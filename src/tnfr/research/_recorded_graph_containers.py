"""Strict finite tagged containers for detached graph-state records."""

from __future__ import annotations

from collections import deque
from fractions import Fraction
from math import isfinite

from ..mathematics.epi import BEPIElement
from ..types import require_finite_real_scalar_epi, serialize_bepi


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def decode_recorded_container(value):
    """Decode only finite scalar tags and declared built-in containers.

    Lists, tuples and bounded deques remain distinct. A mapping is rebuilt
    only after checking duplicate keys, including Python-equal keys. Physical
    scalars must subsequently pass their own admission; metadata booleans are
    not silently promoted into physical values here.
    """
    if value is None or type(value) in (bool, int, str, Fraction):
        return value
    _require(type(value) is dict and "type" in value, "tagged value required")
    kind = value["type"]
    if kind == "binary64":
        _require(set(value) == {"type", "hex"}, "invalid binary64 record")
        _require(type(value["hex"]) is str, "binary64 hex must be text")
        result = float.fromhex(value["hex"])
        _require(
            isfinite(result) and result.hex() == value["hex"], "invalid finite binary64"
        )
        return result
    if kind == "builtins.complex":
        _require(set(value) == {"type", "real", "imaginary"}, "invalid complex record")
        real, imaginary = (
            decode_recorded_container(value[key]) for key in ("real", "imaginary")
        )
        _require(
            type(real) is type(imaginary) is float,
            "complex components must be binary64",
        )
        return complex(real, imaginary)
    _require(
        kind
        in {"builtins.dict", "builtins.list", "builtins.tuple", "collections.deque"},
        "unsupported recorded type",
    )
    expected = (
        {"type", "items", "maxlen"}
        if kind == "collections.deque"
        else {"type", "items"}
    )
    _require(set(value) == expected, "invalid container fields")
    items = value["items"]
    _require(type(items) in (tuple, list), "container items must be ordered")
    if kind == "builtins.dict":
        result = {}
        for row in items:
            _require(
                type(row) in (tuple, list) and len(row) == 2, "invalid mapping pair"
            )
            key, item = row
            # Study metadata and graph-state field names use primitive keys.
            _require(type(key) in (str, int), "unsupported mapping key")
            _require(key not in result, "duplicate recorded mapping key")
            result[key] = decode_recorded_container(item)
        return result
    decoded = tuple(decode_recorded_container(item) for item in items)
    if kind == "builtins.list":
        return list(decoded)
    if kind == "builtins.tuple":
        return decoded
    maxlen = value["maxlen"]
    _require(
        maxlen is None or (type(maxlen) is int and maxlen >= len(decoded)),
        "deque maxlen would discard recorded history",
    )
    return deque(decoded, maxlen=maxlen)


def encode_recorded_container(value):
    """Project finite primitive state; admit BEPI through its signed-form owner."""
    if value is None or type(value) in (bool, int, str, Fraction):
        return value
    if type(value) is float:
        _require(isfinite(value), "nonfinite recorded scalar")
        return {"type": "binary64", "hex": value.hex()}
    if type(value) is complex:
        return {
            "type": "builtins.complex",
            "real": encode_recorded_container(value.real),
            "imaginary": encode_recorded_container(value.imag),
        }
    if type(value) is BEPIElement:
        require_finite_real_scalar_epi(value, "recorded EPI")
        return encode_recorded_container(serialize_bepi(value))
    if type(value) is dict:
        _require(
            all(type(key) in (str, int) for key in value), "unsupported mapping key"
        )
        return {
            "type": "builtins.dict",
            "items": [
                [key, encode_recorded_container(item)] for key, item in value.items()
            ],
        }
    _require(type(value) in (tuple, list, deque), "unsupported recorded value")
    result = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "items": [encode_recorded_container(item) for item in value],
    }
    if type(value) is deque:
        result["maxlen"] = value.maxlen
    return result
