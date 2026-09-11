"""Internal immutable signatures for proof-bearing structural values.

The returned tree contains only immutable built-in scalars, bytes, and tuples.
It preserves IEEE-754 bits, snapshots observable object state, represents
cycles by traversal-local references, and never invokes user ``repr`` or
``hash`` implementations. Unreadable structural state is rejected so proof
validation fails closed.
"""

from __future__ import annotations

import struct
from collections import defaultdict, deque
from collections.abc import Iterable
from fractions import Fraction
from functools import partial
from random import Random
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    CellType,
    FunctionType,
    GetSetDescriptorType,
    MappingProxyType,
    MemberDescriptorType,
    MethodType,
    ModuleType,
)
from typing import Any

from ..mathematics.unified_numerical import np

__all__ = (
    "StructuralSignatureError",
    "binary64_vectors_are_identical",
    "proof_stamps_are_identical",
    "structural_object_state_signature",
    "structural_proof_signature",
)


class StructuralSignatureError(ValueError):
    """Raised when structural state cannot be captured without user code."""


_TYPE_DESCRIPTORS = type.__dict__
_TYPE_DICT = _TYPE_DESCRIPTORS["__dict__"]
_TYPE_MODULE = _TYPE_DESCRIPTORS["__module__"]
_TYPE_QUALNAME = _TYPE_DESCRIPTORS["__qualname__"]
_TYPE_NAME = _TYPE_DESCRIPTORS["__name__"]
_TYPE_MRO = _TYPE_DESCRIPTORS["__mro__"]
_MISSING = object()
_RANDOM_GETSTATE = Random.getstate
_DEFAULTDICT_FACTORY = defaultdict.__dict__["default_factory"]
_METHOD_FUNCTION = MethodType.__dict__["__func__"]
_METHOD_RECEIVER = MethodType.__dict__["__self__"]
_BUILTIN_METHOD_RECEIVER = BuiltinMethodType.__dict__["__self__"]
_BUILTIN_METHOD_MODULE = BuiltinMethodType.__dict__["__module__"]
_BUILTIN_METHOD_NAME = BuiltinMethodType.__dict__["__name__"]
_BUILTIN_METHOD_QUALNAME = BuiltinMethodType.__dict__["__qualname__"]
_BUILTIN_METHOD_TEXT_SIGNATURE = BuiltinMethodType.__dict__["__text_signature__"]
_PARTIAL_FUNCTION = partial.__dict__["func"]
_PARTIAL_ARGUMENTS = partial.__dict__["args"]
_PARTIAL_KEYWORDS = partial.__dict__["keywords"]
_CELL_CONTENTS = CellType.__dict__["cell_contents"]
_FUNCTION_STATE_NAMES = (
    "__annotations__",
    "__code__",
    "__defaults__",
    "__doc__",
    "__kwdefaults__",
    "__module__",
    "__name__",
    "__qualname__",
    "__type_params__",
)

if np is not None:
    _NDARRAY_DESCRIPTORS = np.ndarray.__dict__
    _NDARRAY_DTYPE = _NDARRAY_DESCRIPTORS["dtype"]
    _NDARRAY_FLAGS = _NDARRAY_DESCRIPTORS["flags"]
    _NDARRAY_FLAT = _NDARRAY_DESCRIPTORS["flat"]
    _NDARRAY_SHAPE = _NDARRAY_DESCRIPTORS["shape"]
    _NDARRAY_STRIDES = _NDARRAY_DESCRIPTORS["strides"]
    _NDARRAY_TOBYTES = _NDARRAY_DESCRIPTORS["tobytes"]
    _DTYPE_DESCRIPTORS = np.dtype.__dict__
    _DTYPE_DESCR = _DTYPE_DESCRIPTORS["descr"]
    _DTYPE_HASOBJECT = _DTYPE_DESCRIPTORS["hasobject"]
    _DTYPE_METADATA = _DTYPE_DESCRIPTORS["metadata"]
    _DTYPE_STR = _DTYPE_DESCRIPTORS["str"]


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


def _slot_state(
    value: Any,
    *,
    seen: dict[int, tuple[int, Any]],
    opaque_ids: frozenset[int],
    identity_sensitive: bool,
) -> tuple[Any, ...]:
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
                slot_value = (
                    "value",
                    _freeze(
                        observed,
                        seen=seen,
                        opaque_ids=opaque_ids,
                        identity_sensitive=identity_sensitive,
                    ),
                )
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


def _instance_state(
    value: Any,
    *,
    seen: dict[int, tuple[int, Any]],
    opaque_ids: frozenset[int],
    identity_sensitive: bool,
) -> tuple[Any, ...]:
    namespace = _instance_namespace(value)
    namespace_state = (
        ("no-namespace",)
        if namespace is None
        else (
            "namespace",
            id(namespace),
            _freeze(
                namespace,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
        )
    )
    return (
        namespace_state,
        _slot_state(
            value,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=identity_sensitive,
        ),
    )


def _custom_container_state(
    value: Any,
    builtin: type[Any],
    *,
    seen: dict[int, tuple[int, Any]],
    opaque_ids: frozenset[int],
    identity_sensitive: bool,
) -> tuple[Any, ...]:
    if type(value) is builtin:
        return ("builtin",)
    return (
        "subclass",
        id(value),
        _instance_state(
            value,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=identity_sensitive,
        ),
    )


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


def _unordered_sort_records(
    values: Iterable[Any],
    *,
    seen: dict[int, tuple[int, Any]],
    opaque_ids: frozenset[int],
    identity_sensitive: bool,
) -> tuple[tuple[bytes, int, Any, Any], ...]:
    """Order unordered values without losing distinct equal-token identities.

    Non-reflexive built-ins such as separate bit-identical NaNs may coexist as
    mapping keys or set members even though their structural value tokens are
    identical.  Their process-local identity is therefore the only safe
    deterministic tie-breaker.  It is exposed in the final token only for such
    a collision, where identity is part of the container's observable key or
    membership semantics.
    """

    records: list[tuple[bytes, int, Any, Any]] = []
    for value in values:
        token = _freeze(
            value,
            seen=seen.copy(),
            opaque_ids=opaque_ids,
            identity_sensitive=identity_sensitive,
        )
        records.append((_canonical_bytes(token), id(value), value, token))
    records.sort(key=lambda record: (record[0], record[1]))
    return tuple(records)


def _unordered_value_token(
    value: Any,
    *,
    sort_token: Any,
    token_occurrences: dict[bytes, int],
    canonical_token: bytes,
    seen: dict[int, tuple[int, Any]],
    opaque_ids: frozenset[int],
    identity_sensitive: bool,
) -> Any:
    """Freeze one sorted member and retain identity only on token collision."""

    token = _freeze(
        value,
        seen=seen,
        opaque_ids=opaque_ids,
        identity_sensitive=identity_sensitive,
    )
    if token_occurrences[canonical_token] == 1:
        return token
    return ("distinct-equal-token", sort_token, id(value), token)


def _mapping_proxy_contents_signature(
    value: MappingProxyType,
    *,
    seen: dict[int, tuple[int, Any]],
    opaque_ids: frozenset[int],
    identity_sensitive: bool,
) -> tuple[Any, ...]:
    """Freeze mapping-proxy contents without signing its wrapper identity."""

    try:
        pairs = list(MappingProxyType.items(value))
    except BaseException as exc:
        raise _unreadable("mapping proxy", exc) from None
    ordered_keys = _unordered_sort_records(
        (key for key, _mapped in pairs),
        seen=seen,
        opaque_ids=opaque_ids,
        identity_sensitive=identity_sensitive,
    )
    mapped_by_identity = {id(key): mapped for key, mapped in pairs}
    occurrences: dict[bytes, int] = {}
    for canonical, _identity, _key, _token in ordered_keys:
        occurrences[canonical] = occurrences.get(canonical, 0) + 1
    return tuple(
        (
            _unordered_value_token(
                key,
                sort_token=sort_token,
                token_occurrences=occurrences,
                canonical_token=canonical,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
            _freeze(
                mapped_by_identity[key_identity],
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
        )
        for canonical, key_identity, key, sort_token in ordered_keys
    )


def _identity_token(
    identity: int,
    *,
    identity_sensitive: bool,
) -> tuple[tuple[str, int], ...]:
    if not identity_sensitive:
        return ()
    return (("object-identity", identity),)


def _track_reference(
    value: Any,
    seen: dict[int, tuple[int, Any]],
) -> int | None:
    """Remember a live object strongly and return its prior traversal index."""

    identity = id(value)
    record = seen.get(identity)
    if record is not None and record[1] is value:
        return record[0]
    seen[identity] = (len(seen), value)
    return None


def _ndarray_signature(
    value: Any,
    *,
    seen: dict[int, tuple[int, Any]],
    opaque_ids: frozenset[int],
    identity_sensitive: bool,
) -> tuple[Any, ...]:
    """Capture NumPy array layout and contents through base descriptors."""

    kind = type(value)
    try:
        dtype = _descriptor_get(_NDARRAY_DTYPE, value, kind)
        shape = _descriptor_get(_NDARRAY_SHAPE, value, kind)
        strides = _descriptor_get(_NDARRAY_STRIDES, value, kind)
        flags = _descriptor_get(_NDARRAY_FLAGS, value, kind)
        flags_kind = type(flags)
        writeable_descriptor = flags_kind.__dict__["writeable"]
        writeable = _descriptor_get(writeable_descriptor, flags, flags_kind)
        dtype_kind = type(dtype)
        dtype_text = _descriptor_get(_DTYPE_STR, dtype, dtype_kind)
        dtype_description = _descriptor_get(_DTYPE_DESCR, dtype, dtype_kind)
        dtype_metadata = _descriptor_get(_DTYPE_METADATA, dtype, dtype_kind)
        has_objects = _descriptor_get(_DTYPE_HASOBJECT, dtype, dtype_kind)
        if has_objects:
            flat = _descriptor_get(_NDARRAY_FLAT, value, kind)
            contents = (
                "objects",
                tuple(
                    _freeze(
                        item,
                        seen=seen,
                        opaque_ids=opaque_ids,
                        identity_sensitive=identity_sensitive,
                    )
                    for item in flat
                ),
            )
        else:
            contents = ("bytes", _NDARRAY_TOBYTES(value, order="A"))
    except BaseException as exc:
        raise _unreadable("NumPy array state", exc) from None
    return (
        "ndarray",
        *_type_identity(value),
        *_identity_token(
            id(value),
            identity_sensitive=identity_sensitive,
        ),
        _custom_container_state(
            value,
            np.ndarray,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=identity_sensitive,
        ),
        _freeze(
            shape,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=False,
        ),
        _freeze(
            strides,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=False,
        ),
        _freeze(
            dtype_text,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=False,
        ),
        _freeze(
            dtype_description,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=False,
        ),
        (
            ("none",)
            if dtype_metadata is None
            else (
                "dtype-metadata",
                _mapping_proxy_contents_signature(
                    dtype_metadata,
                    seen=seen,
                    opaque_ids=opaque_ids,
                    identity_sensitive=identity_sensitive,
                ),
            )
        ),
        _freeze(
            writeable,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=False,
        ),
        contents,
    )


def _dtype_signature(
    value: Any,
    *,
    seen: dict[int, tuple[int, Any]],
    opaque_ids: frozenset[int],
    identity_sensitive: bool,
) -> tuple[Any, ...]:
    """Capture a NumPy dtype, including mutable objects in its metadata."""

    reference = _track_reference(value, seen)
    if reference is not None:
        return ("reference", reference)
    try:
        kind = type(value)
        dtype_text = _descriptor_get(_DTYPE_STR, value, kind)
        dtype_description = _descriptor_get(_DTYPE_DESCR, value, kind)
        dtype_metadata = _descriptor_get(_DTYPE_METADATA, value, kind)
        has_objects = _descriptor_get(_DTYPE_HASOBJECT, value, kind)
    except BaseException as exc:
        raise _unreadable("NumPy dtype state", exc) from None
    return (
        "numpy-dtype",
        *_type_identity(value),
        *_identity_token(id(value), identity_sensitive=identity_sensitive),
        _freeze(
            dtype_text,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=False,
        ),
        _freeze(
            dtype_description,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=False,
        ),
        (
            ("none",)
            if dtype_metadata is None
            else (
                "dtype-metadata",
                _mapping_proxy_contents_signature(
                    dtype_metadata,
                    seen=seen,
                    opaque_ids=opaque_ids,
                    identity_sensitive=identity_sensitive,
                ),
            )
        ),
        _freeze(
            has_objects,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=False,
        ),
    )


def _function_signature(
    value: FunctionType,
    *,
    seen: dict[int, tuple[int, Any]],
    opaque_ids: frozenset[int],
    identity_sensitive: bool,
) -> tuple[Any, ...]:
    """Capture mutable function bindings without traversing module globals."""

    intrinsic: list[tuple[str, Any]] = []
    try:
        for name in _FUNCTION_STATE_NAMES:
            descriptor = FunctionType.__dict__.get(name)
            if descriptor is not None:
                intrinsic.append(
                    (name, _descriptor_get(descriptor, value, FunctionType))
                )
        closure_descriptor = FunctionType.__dict__["__closure__"]
        closure = _descriptor_get(closure_descriptor, value, FunctionType)
        cells: list[tuple[int, str, Any]] = []
        for cell in closure or ():
            try:
                cell_value = _descriptor_get(_CELL_CONTENTS, cell, CellType)
            except ValueError:
                cells.append((id(cell), "empty", None))
            else:
                cells.append((id(cell), "value", cell_value))
    except BaseException as exc:
        raise _unreadable("function state", exc) from None
    return (
        "function",
        *_type_identity(value),
        id(value),
        _instance_state(
            value,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=identity_sensitive,
        ),
        tuple(
            (
                name,
                _freeze(
                    item,
                    seen=seen,
                    opaque_ids=opaque_ids,
                    identity_sensitive=identity_sensitive,
                ),
            )
            for name, item in intrinsic
        ),
        tuple(
            (
                cell_id,
                state,
                _freeze(
                    item,
                    seen=seen,
                    opaque_ids=opaque_ids,
                    identity_sensitive=identity_sensitive,
                ),
            )
            for cell_id, state, item in cells
        ),
    )


def _builtin_callable_signature(
    value: BuiltinFunctionType,
    *,
    seen: dict[int, tuple[int, Any]],
    opaque_ids: frozenset[int],
    identity_sensitive: bool,
) -> tuple[Any, ...]:
    """Capture a built-in callable's definition and bound receiver safely."""

    try:
        receiver = _descriptor_get(
            _BUILTIN_METHOD_RECEIVER,
            value,
            BuiltinMethodType,
        )
        metadata = tuple(
            _descriptor_get(descriptor, value, BuiltinMethodType)
            for descriptor in (
                _BUILTIN_METHOD_MODULE,
                _BUILTIN_METHOD_NAME,
                _BUILTIN_METHOD_QUALNAME,
                _BUILTIN_METHOD_TEXT_SIGNATURE,
            )
        )
    except BaseException as exc:
        raise _unreadable("built-in callable state", exc) from None

    if receiver is None:
        receiver_signature: Any = ("unbound",)
    elif _is_subtype(type(receiver), ModuleType):
        # Module globals are ambient runtime state, not callable-owned state.
        # The module identity plus the callable metadata distinguishes its
        # binding without recursively traversing a potentially huge namespace.
        receiver_signature = (
            "module-receiver",
            *_type_identity(receiver),
            id(receiver),
        )
    else:
        receiver_signature = _freeze(
            receiver,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=identity_sensitive,
        )

    return (
        "built-in-callable",
        *_type_identity(value),
        tuple(
            _freeze(
                item,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=False,
            )
            for item in metadata
        ),
        receiver_signature,
    )


def _freeze(
    value: Any,
    *,
    seen: dict[int, tuple[int, Any]],
    opaque_ids: frozenset[int],
    identity_sensitive: bool,
) -> Any:
    identity = id(value)
    if identity in opaque_ids:
        return ("opaque-reference", *_type_identity(value), identity)
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
    if _is_subtype(type(value), bytearray):
        reference = _track_reference(value, seen)
        if reference is not None:
            return ("reference", reference)
        return (
            "bytearray",
            *_type_identity(value),
            *_identity_token(
                identity,
                identity_sensitive=identity_sensitive,
            ),
            _custom_container_state(
                value,
                bytearray,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
            bytes(tuple(bytearray.__iter__(value))),
        )
    if type(value) is memoryview:
        reference = _track_reference(value, seen)
        if reference is not None:
            return ("reference", reference)
        try:
            return (
                "memoryview",
                *_identity_token(
                    identity,
                    identity_sensitive=identity_sensitive,
                ),
                value.tobytes(),
            )
        except BaseException as exc:
            raise _unreadable("memoryview", exc) from None
    if np is not None and _is_subtype(type(value), np.ndarray):
        reference = _track_reference(value, seen)
        if reference is not None:
            return ("reference", reference)
        return _ndarray_signature(
            value,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=identity_sensitive,
        )
    if np is not None and _is_subtype(type(value), np.dtype):
        return _dtype_signature(
            value,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=identity_sensitive,
        )
    if _is_subtype(type(value), Random):
        reference = _track_reference(value, seen)
        if reference is not None:
            return ("reference", reference)
        try:
            state = _RANDOM_GETSTATE(value)
        except BaseException as exc:
            raise _unreadable("random generator state", exc) from None
        return (
            "random.Random",
            *_type_identity(value),
            *_identity_token(
                identity,
                identity_sensitive=identity_sensitive,
            ),
            _instance_state(
                value,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
            _freeze(
                state,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=False,
            ),
        )
    if _is_subtype(type(value), partial):
        reference = _track_reference(value, seen)
        if reference is not None:
            return ("reference", reference)
        try:
            function = _descriptor_get(_PARTIAL_FUNCTION, value, partial)
            arguments = _descriptor_get(_PARTIAL_ARGUMENTS, value, partial)
            keywords = _descriptor_get(_PARTIAL_KEYWORDS, value, partial)
        except BaseException as exc:
            raise _unreadable("partial state", exc) from None
        return (
            "functools.partial",
            *_type_identity(value),
            id(value),
            _instance_state(
                value,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
            _freeze(
                function,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
            _freeze(
                arguments,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
            _freeze(
                keywords,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
        )
    if _is_subtype(type(value), MethodType):
        reference = _track_reference(value, seen)
        if reference is not None:
            return ("reference", reference)
        try:
            function = _descriptor_get(_METHOD_FUNCTION, value, MethodType)
            receiver = _descriptor_get(_METHOD_RECEIVER, value, MethodType)
        except BaseException as exc:
            raise _unreadable("bound method state", exc) from None
        return (
            "bound-method",
            _freeze(
                function,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
            _freeze(
                receiver,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
        )
    if type(value) is FunctionType:
        reference = _track_reference(value, seen)
        if reference is not None:
            return ("reference", reference)
        return _function_signature(
            value,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=identity_sensitive,
        )
    # CPython exposes built-in functions and bound built-in methods through
    # the same ``builtin_function_or_method`` runtime type.
    if _is_subtype(type(value), BuiltinFunctionType):
        reference = _track_reference(value, seen)
        if reference is not None:
            return ("reference", reference)
        return _builtin_callable_signature(
            value,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=identity_sensitive,
        )
    if _is_subtype(type(value), type):
        return ("type", _class_identity(value))

    reference = _track_reference(value, seen)
    if reference is not None:
        return ("reference", reference)

    kind = type(value)
    if _is_subtype(kind, tuple):
        items = _materialize(value, tuple, "tuple")
        return (
            "tuple",
            *_type_identity(value),
            *_identity_token(
                identity,
                identity_sensitive=identity_sensitive,
            ),
            _custom_container_state(
                value,
                tuple,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
            tuple(
                _freeze(
                    item,
                    seen=seen,
                    opaque_ids=opaque_ids,
                    identity_sensitive=identity_sensitive,
                )
                for item in items
            ),
        )
    if _is_subtype(kind, list):
        items = _materialize(value, list, "list")
        return (
            "list",
            *_type_identity(value),
            *_identity_token(
                identity,
                identity_sensitive=identity_sensitive,
            ),
            _custom_container_state(
                value,
                list,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
            tuple(
                _freeze(
                    item,
                    seen=seen,
                    opaque_ids=opaque_ids,
                    identity_sensitive=identity_sensitive,
                )
                for item in items
            ),
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
            *_identity_token(
                identity,
                identity_sensitive=identity_sensitive,
            ),
            _custom_container_state(
                value,
                deque,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
            _freeze(
                maxlen,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
            tuple(
                _freeze(
                    item,
                    seen=seen,
                    opaque_ids=opaque_ids,
                    identity_sensitive=identity_sensitive,
                )
                for item in items
            ),
        )
    if _is_subtype(kind, dict):
        try:
            pairs = list(dict.items(value))
        except BaseException as exc:
            raise _unreadable("mapping", exc) from None
        ordered_keys = _unordered_sort_records(
            (key for key, _mapped in pairs),
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=identity_sensitive,
        )
        mapped_by_identity = {
            id(key): mapped for key, mapped in pairs
        }
        occurrences: dict[bytes, int] = {}
        for canonical, _identity, _key, _token in ordered_keys:
            occurrences[canonical] = occurrences.get(canonical, 0) + 1
        frozen_items = tuple(
            (
                _unordered_value_token(
                    key,
                    sort_token=sort_token,
                    token_occurrences=occurrences,
                    canonical_token=canonical,
                    seen=seen,
                    opaque_ids=opaque_ids,
                    identity_sensitive=identity_sensitive,
                ),
                _freeze(
                    mapped_by_identity[key_identity],
                    seen=seen,
                    opaque_ids=opaque_ids,
                    identity_sensitive=identity_sensitive,
                ),
            )
            for canonical, key_identity, key, sort_token in ordered_keys
        )
        default_factory = (
            (
                "default-factory",
                _freeze(
                    _descriptor_get(_DEFAULTDICT_FACTORY, value, kind),
                    seen=seen,
                    opaque_ids=opaque_ids,
                    identity_sensitive=identity_sensitive,
                ),
            )
            if _is_subtype(kind, defaultdict)
            else ("no-default-factory",)
        )
        return (
            "mapping",
            *_type_identity(value),
            *_identity_token(
                identity,
                identity_sensitive=identity_sensitive,
            ),
            _custom_container_state(
                value,
                dict,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
            default_factory,
            frozen_items,
        )
    if type(value) is MappingProxyType:
        return (
            "mapping-proxy",
            *_identity_token(
                identity,
                identity_sensitive=identity_sensitive,
            ),
            _mapping_proxy_contents_signature(
                value,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
        )
    if _is_subtype(kind, set) or _is_subtype(kind, frozenset):
        builtin = set if _is_subtype(kind, set) else frozenset
        items = _materialize(value, builtin, "set")
        ordered = _unordered_sort_records(
            items,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=identity_sensitive,
        )
        occurrences = {}
        for canonical, _identity, _item, _token in ordered:
            occurrences[canonical] = occurrences.get(canonical, 0) + 1
        frozen_items = tuple(
            _unordered_value_token(
                item,
                sort_token=sort_token,
                token_occurrences=occurrences,
                canonical_token=canonical,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            )
            for canonical, _identity, item, sort_token in ordered
        )
        return (
            "set-like",
            *_type_identity(value),
            *_identity_token(
                identity,
                identity_sensitive=identity_sensitive,
            ),
            _custom_container_state(
                value,
                builtin,
                seen=seen,
                opaque_ids=opaque_ids,
                identity_sensitive=identity_sensitive,
            ),
            frozen_items,
        )
    namespace = _instance_namespace(value)
    slots = _slot_state(
        value,
        seen=seen,
        opaque_ids=opaque_ids,
        identity_sensitive=identity_sensitive,
    )
    if namespace is not None or slots:
        return (
            "object-state",
            *_type_identity(value),
            identity,
            ("no-namespace",)
            if namespace is None
            else (
                "namespace",
                id(namespace),
                _freeze(
                    namespace,
                    seen=seen,
                    opaque_ids=opaque_ids,
                    identity_sensitive=identity_sensitive,
                ),
            ),
            slots,
        )
    return ("opaque-identity", *_type_identity(value), identity)


def binary64_vectors_are_identical(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> bool:
    """Compare binary64 vectors bit for bit, preserving signed zero and NaNs."""

    return bool(
        len(left) == len(right)
        and all(
            struct.pack(">d", first) == struct.pack(">d", second)
            for first, second in zip(left, right, strict=True)
        )
    )


def structural_proof_signature(
    value: Any,
    *,
    opaque_references: Iterable[Any] = (),
    identity_sensitive: bool = False,
    _retained_references: list[Any] | None = None,
) -> Any:
    """Return a detached immutable, bit-faithful structural signature.

    Traversal-local reference markers preserve cycles and shared references.
    Built-in immutable labels remain value-based. Custom objects carry a
    process-local identity token plus readable state; hostile ``repr`` and
    ``hash`` implementations are never invoked. Explicit opaque references
    retain only their safe type and process-local identity, allowing callers to
    sign an owning object without recursively duplicating separately protected
    external state. ``identity_sensitive=True`` additionally records container
    identities so equal-value rebinding remains observable while the original
    bindings remain live. Runtime guards can supply ``_retained_references`` to
    extend that lifetime through their comparison boundary.
    """

    # Keep every opaque referent alive for the full traversal. Otherwise an
    # ephemeral input iterable can release an object, CPython can reuse its
    # address for a freshly materialized descriptor value, and the new value
    # would be mistaken for an opaque reference.
    opaque_values = tuple(opaque_references)
    opaque_ids = frozenset(id(item) for item in opaque_values)
    seen: dict[int, tuple[int, Any]] = {}
    signature = _freeze(
        value,
        seen=seen,
        opaque_ids=opaque_ids,
        identity_sensitive=identity_sensitive,
    )
    if _retained_references is not None:
        _retained_references.extend(opaque_values)
        _retained_references.extend(record[1] for record in seen.values())
    return signature


def proof_stamps_are_identical(observed: Any, expected: Any) -> bool:
    """Compare two proof stamps without invoking caller-owned protocols.

    Proof stamps are exact built-in tuples.  Either value may nevertheless have
    been replaced through ``object.__setattr__`` after construction.  The
    comparator walks the closed immutable token grammar directly; it never
    calls generic equality, truth conversion or a caller-owned descriptor.  A
    direct walk also avoids recursively re-encoding already signed nested
    evidence, whose token tree can be large.  Unknown values and every
    exception category fail closed.
    """

    if type(observed) is not tuple or type(expected) is not tuple:
        return False
    try:
        pending: list[tuple[Any, Any]] = [(observed, expected)]
        while pending:
            left, right = pending.pop()
            kind = type(left)
            if kind is not type(right):
                return False
            if kind is tuple:
                if tuple.__len__(left) != tuple.__len__(right):
                    return False
                pending.extend(
                    zip(tuple.__iter__(left), tuple.__iter__(right), strict=True)
                )
            elif kind is type(None):
                continue
            elif kind is bool:
                if left is not right:
                    return False
            elif kind is int:
                if not int.__eq__(left, right):
                    return False
            elif kind is float:
                if struct.pack(">d", left) != struct.pack(">d", right):
                    return False
            elif kind is complex:
                if (
                    struct.pack(">d", left.real) != struct.pack(">d", right.real)
                    or struct.pack(">d", left.imag)
                    != struct.pack(">d", right.imag)
                ):
                    return False
            elif kind is Fraction:
                if (
                    left.numerator != right.numerator
                    or left.denominator != right.denominator
                ):
                    return False
            elif kind is str:
                if not str.__eq__(left, right):
                    return False
            elif kind is bytes:
                if not bytes.__eq__(left, right):
                    return False
            else:
                return False
        return True
    except BaseException:
        return False


def structural_object_state_signature(
    value: Any,
    *,
    opaque_references: Iterable[Any] = (),
    identity_sensitive: bool = False,
    _retained_references: list[Any] | None = None,
) -> Any:
    """Return a signature of an object's identity and instance-owned state.

    Container contents are deliberately excluded.  This makes the helper useful
    when a caller permits selected mapping entries to change but must still
    detect mutations to a custom mapping's ``__dict__`` or any slot declared
    anywhere in its MRO.  Slots are read through their owner descriptors, so a
    subclass redeclaring a slot name cannot hide the base-class storage.
    Explicit opaque references have the same identity-only semantics as in
    :func:`structural_proof_signature`. ``identity_sensitive=True`` also
    distinguishes equal-value mutable bindings while the original bindings
    remain live, as they do throughout the runtime guards that use this helper.
    """

    if _is_subtype(type(value), type):
        return ("type", _class_identity(value))
    seen = {id(value): (0, value)}
    opaque_values = tuple(opaque_references)
    opaque_ids = frozenset(id(item) for item in opaque_values)
    signature = (
        "object-own-state",
        *_type_identity(value),
        id(value),
        _instance_state(
            value,
            seen=seen,
            opaque_ids=opaque_ids,
            identity_sensitive=identity_sensitive,
        ),
    )
    if _retained_references is not None:
        _retained_references.extend(opaque_values)
        _retained_references.extend(record[1] for record in seen.values())
    return signature
