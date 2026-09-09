"""Adversarial tests for internal proof-value signatures."""

from __future__ import annotations

import struct

import pytest

from tnfr.utils._structural_signature import (
    StructuralSignatureError,
    structural_proof_signature,
)


def _assert_immutable_tree(value: object) -> None:
    assert type(value) in {bool, bytes, int, str, tuple}
    if type(value) is tuple:
        for item in value:
            _assert_immutable_tree(item)


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
