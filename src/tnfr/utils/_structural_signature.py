"""Internal immutable signatures for proof-bearing structural values.

The returned tree contains only immutable built-in scalars, bytes, and tuples.
It preserves IEEE-754 bits, snapshots observable object state, represents
cycles by traversal-local references, and never invokes user ``repr`` or
``hash`` implementations. Unreadable structural state is rejected so proof
validation fails closed.
"""

from __future__ import annotations

import struct
from collections import deque
from fractions import Fraction
from types import GetSetDescriptorType, MappingProxyType, MemberDescriptorType
from typing import Any

__all__ = ("StructuralSignatureError", "structural_proof_signature")


class StructuralSignatureError(ValueError):
    """Raised when structural state cannot be captured without user code."""


_TYPE_DESCRIPTORS = type.__dict__
_TYPE_DICT = _TYPE_DESCRIPTORS["__dict__"]
_TYPE_MODULE = _TYPE_DESCRIPTORS["__module__"]
_TYPE_QUALNAME = _TYPE_DESCRIPTORS["__qualname__"]
_TYPE_NAME = _TYPE_DESCRIPTORS["__name__"]
_TYPE_MRO = _TYPE_DESCRIPTORS["__mro__"]
_MISSING = object()


def _descriptor_get(descriptor: Any, value: Any, owner: type[Any]) -> Any:
    """Invoke only a known built-in descriptor implementation."""

    return type(descriptor).__get__(descriptor, value, owner)


def _class_text_token(kind: type[Any], name: str) -> str:
    descriptor = {
        "__module__": _TYPE_MODULE,
        "__qualname__": _TYPE_QUALNAME,
        "__name__": _TYPE_NAME,
    }[name]
    metaclass = type(kind)
    if metaclass is not type:
        for owner in _class_mro(metaclass):
            if owner is type:
                break
            shadow = _class_namespace(owner).get(name, _MISSING)
            if shadow is not _MISSING and type(shadow) is not str:
                raise StructuralSignatureError(
                    f"class {name} is shadowed by custom metadata"
                )
    try:
        value = _descriptor_get(descriptor, kind, type(kind))
    except BaseException as exc:
        raise _unreadable(f"class {name}", exc) from None
    if type(value) is not str:
        raise StructuralSignatureError(f"class {name} must be a string")
    return value


def _class_identity(kind: type[Any]) -> tuple[Any, ...]:
    return (
        "class",
        _class_text_token(kind, "__module__"),
        _class_text_token(kind, "__qualname__"),
        id(kind),
    )


def _type_identity(value: Any) -> tuple[Any, ...]:
    return _class_identity(type(value))


def _unreadable(label: str, exc: BaseException) -> StructuralSignatureError:
    del exc
    return StructuralSignatureError(f"{label} is unreadable")


def _class_mro(kind: type[Any]) -> tuple[type[Any], ...]:
    try:
        owners = _descriptor_get(_TYPE_MRO, kind, type(kind))
    except BaseException as exc:
        raise _unreadable("class MRO", exc) from None
    if type(owners) is not tuple:
        raise StructuralSignatureError("class MRO must be a tuple")
    return owners


def _is_subtype(kind: type[Any], base: type[Any]) -> bool:
    return any(owner is base for owner in _class_mro(kind))


def _class_namespace(kind: type[Any]) -> MappingProxyType:
    try:
        namespace = _descriptor_get(_TYPE_DICT, kind, type(kind))
    except BaseException as exc:
        raise _unreadable("class namespace", exc) from None
    if type(namespace) is not MappingProxyType:
        raise StructuralSignatureError("class namespace must be a mapping proxy")
    return namespace


def _slot_storage_name(owner: type[Any], declared: str) -> str:
    if declared.startswith("__") and not declared.endswith("__"):
        raw_name = _class_text_token(owner, "__name__")
        return f"_{raw_name.lstrip('_')}{declared}"
    return declared


def _slot_state(value: Any, *, seen: dict[int, int]) -> tuple[Any, ...]:
    state: list[Any] = []
    owners = _class_mro(type(value))
    for owner in owners:
        owner_namespace = _class_namespace(owner)
        declared_slots = owner_namespace.get("__slots__", ())
        if type(declared_slots) is str:
            slot_names = (declared_slots,)
        elif type(declared_slots) is tuple:
            slot_names = tuple(tuple.__iter__(declared_slots))
        elif type(declared_slots) is list:
            slot_names = tuple(list.__iter__(declared_slots))
        else:
            raise StructuralSignatureError(
                "slot declaration must be str, tuple, or list"
            )
        for declared in slot_names:
            if type(declared) is not str:
                raise StructuralSignatureError("slot names must be strings")
            if declared in ("__dict__", "__weakref__"):
                continue
            storage_name = _slot_storage_name(owner, declared)
            descriptor = owner_namespace.get(storage_name)
            if type(descriptor) is not MemberDescriptorType:
                raise StructuralSignatureError(
                    "slot descriptor is not a built-in member"
                )
            try:
                observed = _descriptor_get(descriptor, value, type(value))
            except AttributeError:
                slot_value = ("unset",)
            except BaseException as exc:
                raise _unreadable("slot value", exc) from None
            else:
                slot_value = ("value", _freeze(observed, seen=seen))
            state.append(
                (_class_identity(owner), declared, slot_value)
            )
    return tuple(state)


def _materialize(value: Any, builtin: type[Any], label: str) -> tuple[Any, ...]:
    try:
        return tuple(builtin.__iter__(value))
    except BaseException as exc:
        raise _unreadable(label, exc) from None


def _instance_namespace(value: Any) -> dict[Any, Any] | None:
    for owner in _class_mro(type(value)):
        descriptor = _class_namespace(owner).get("__dict__")
        if descriptor is None:
            continue
        if type(descriptor) is not GetSetDescriptorType:
            raise StructuralSignatureError(
                "instance namespace descriptor is not a built-in getset"
            )
        try:
            namespace = _descriptor_get(descriptor, value, type(value))
        except BaseException as exc:
            raise _unreadable("object namespace", exc) from None
        if type(namespace) is not dict:
            raise StructuralSignatureError("object namespace must be a built-in dict")
        return namespace
    return None


def _instance_state(value: Any, *, seen: dict[int, int]) -> tuple[Any, ...]:
    namespace = _instance_namespace(value)
    namespace_state = (
        ("no-namespace",)
        if namespace is None
        else _freeze(namespace, seen=seen)
    )
    return (namespace_state, _slot_state(value, seen=seen))


def _custom_container_state(
    value: Any,
    builtin: type[Any],
    *,
    seen: dict[int, int],
) -> tuple[Any, ...]:
    if type(value) is builtin:
        return ("builtin",)
    return ("subclass", id(value), _instance_state(value, seen=seen))


def _canonical_bytes(value: Any) -> bytes:
    """Encode an already-frozen token without invoking user methods."""

    if type(value) is tuple:
        encoded = tuple(_canonical_bytes(item) for item in value)
        return b"t" + b"".join(
            len(item).to_bytes(8, "big") + item for item in encoded
        )
    if type(value) is bool:
        return b"b1" if value else b"b0"
    if type(value) is int:
        payload = str(value).encode("ascii")
        return b"i" + len(payload).to_bytes(8, "big") + payload
    if type(value) is str:
        payload = value.encode("utf-8", errors="surrogatepass")
        return b"s" + len(payload).to_bytes(8, "big") + payload
    if type(value) is bytes:
        return b"y" + len(value).to_bytes(8, "big") + value
    raise StructuralSignatureError("signature contains a non-immutable token")


def _freeze(value: Any, *, seen: dict[int, int]) -> Any:
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        return ("binary64", struct.pack(">d", value))
    if type(value) is complex:
        return (
            "complex128",
            struct.pack(">d", value.real),
            struct.pack(">d", value.imag),
        )
    if type(value) is Fraction:
        return ("fraction", value.numerator, value.denominator)
    if type(value) is str:
        return ("str", value)
    if type(value) is bytes:
        return ("bytes", value)
    if type(value) is bytearray:
        return ("bytearray", bytes(value))
    if type(value) is memoryview:
        try:
            return ("memoryview", value.tobytes())
        except BaseException as exc:
            raise _unreadable("memoryview", exc) from None
    if _is_subtype(type(value), type):
        return ("type", _class_identity(value))

    identity = id(value)
    if identity in seen:
        return ("reference", seen[identity])
    seen[identity] = len(seen)

    kind = type(value)
    if _is_subtype(kind, tuple):
        items = _materialize(value, tuple, "tuple")
        return (
            "tuple",
            *_type_identity(value),
            _custom_container_state(value, tuple, seen=seen),
            tuple(_freeze(item, seen=seen) for item in items),
        )
    if _is_subtype(kind, list):
        items = _materialize(value, list, "list")
        return (
            "list",
            *_type_identity(value),
            _custom_container_state(value, list, seen=seen),
            tuple(_freeze(item, seen=seen) for item in items),
        )
    if _is_subtype(kind, deque):
        items = _materialize(value, deque, "deque")
        try:
            maxlen = _descriptor_get(deque.__dict__["maxlen"], value, kind)
        except BaseException as exc:
            raise _unreadable("deque maxlen", exc) from None
        return (
            "deque",
            *_type_identity(value),
            _custom_container_state(value, deque, seen=seen),
            _freeze(maxlen, seen=seen),
            tuple(_freeze(item, seen=seen) for item in items),
        )
    if _is_subtype(kind, dict):
        try:
            pairs = list(dict.items(value))
        except BaseException as exc:
            raise _unreadable("mapping", exc) from None
        pairs.sort(
            key=lambda pair: _canonical_bytes(
                _freeze(pair[0], seen=seen.copy())
            )
        )
        frozen_items = tuple(
            (_freeze(key, seen=seen), _freeze(mapped, seen=seen))
            for key, mapped in pairs
        )
        return (
            "mapping",
            *_type_identity(value),
            _custom_container_state(value, dict, seen=seen),
            frozen_items,
        )
    if type(value) is MappingProxyType:
        try:
            pairs = list(MappingProxyType.items(value))
        except BaseException as exc:
            raise _unreadable("mapping proxy", exc) from None
        pairs.sort(
            key=lambda pair: _canonical_bytes(
                _freeze(pair[0], seen=seen.copy())
            )
        )
        return (
            "mapping-proxy",
            tuple(
                (_freeze(key, seen=seen), _freeze(mapped, seen=seen))
                for key, mapped in pairs
            ),
        )
    if _is_subtype(kind, set) or _is_subtype(kind, frozenset):
        builtin = set if _is_subtype(kind, set) else frozenset
        items = _materialize(value, builtin, "set")
        ordered = sorted(
            items,
            key=lambda item: _canonical_bytes(_freeze(item, seen=seen.copy())),
        )
        frozen_items = tuple(_freeze(item, seen=seen) for item in ordered)
        return (
            "set-like",
            *_type_identity(value),
            _custom_container_state(value, builtin, seen=seen),
            frozen_items,
        )
    namespace = _instance_namespace(value)
    slots = _slot_state(value, seen=seen)
    if namespace is not None or slots:
        return (
            "object-state",
            *_type_identity(value),
            identity,
            ("no-namespace",)
            if namespace is None
            else _freeze(namespace, seen=seen),
            slots,
        )
    return ("opaque-identity", *_type_identity(value), identity)


def structural_proof_signature(value: Any) -> Any:
    """Return a detached immutable, bit-faithful structural signature.

    Traversal-local reference markers preserve cycles and shared references.
    Built-in immutable labels remain value-based. Custom objects carry a
    process-local identity token plus readable state; hostile ``repr`` and
    ``hash`` implementations are never invoked.
    """

    return _freeze(value, seen={})
