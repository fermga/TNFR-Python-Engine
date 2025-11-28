"""Property-based checks for structured file I/O helpers."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st

from tests.property.strategies import (
    DEFAULT_PROPERTY_MAX_EXAMPLES,
    PROPERTY_TEST_SETTINGS,
    nested_structured_mappings,
)
from tnfr.utils.io import (
    StructuredFileError,
    json_dumps,
    read_structured_file,
    safe_write,
)


def _sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


def _sorted_scalars(values: Iterable[Any]) -> list[Any]:
    return sorted((value for value in values), key=_sort_key)


def _prepare_for_toml(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _prepare_for_toml(inner) for key, inner in value.items()}
    if isinstance(value, set):
        return _sorted_scalars(value)
    if isinstance(value, (list, tuple, deque)):
        return [_prepare_for_toml(item) for item in value]
    return value


def _assert_equivalent(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual.keys()) == set(expected.keys())
        for key, expected_value in expected.items():
            _assert_equivalent(actual[key], expected_value)
        return

    if isinstance(expected, set):
        assert isinstance(actual, (list, tuple, deque, set))
        expected_items = _sorted_scalars(expected)
        actual_items = _sorted_scalars(actual)
        assert len(actual_items) == len(expected_items)
        for actual_value, expected_value in zip(actual_items, expected_items):
            _assert_equivalent(actual_value, expected_value)
        return

    if isinstance(expected, (list, tuple, deque)):
        assert isinstance(actual, (list, tuple, deque))
        actual_items = list(actual)
        expected_items = list(expected)
        assert len(actual_items) == len(expected_items)
        for actual_value, expected_value in zip(actual_items, expected_items):
            _assert_equivalent(actual_value, expected_value)
        return

    assert actual == expected


FILE_IO_PROPERTY_SETTINGS = settings(
    deadline=None,
    max_examples=DEFAULT_PROPERTY_MAX_EXAMPLES,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"Unsupported scalar for TOML serialisation: {value!r}")


def _format_array(values: list[Any]) -> str:
    formatted = ", ".join(_format_scalar(value) for value in values)
    return f"[{formatted}]"


def _toml_dumps(data: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit(prefix: tuple[str, ...], table: dict[str, Any]) -> None:
        scalars: list[tuple[str, Any]] = []
        nested: list[tuple[str, dict[str, Any]]] = []
        for key, value in table.items():
            if isinstance(value, dict):
                nested.append((key, value))
            else:
                scalars.append((key, value))

        if prefix:
            lines.append(f"[{'.'.join(prefix)}]")

        for key, value in sorted(scalars):
            rendered = _format_array(value) if isinstance(value, list) else _format_scalar(value)
            lines.append(f"{key} = {rendered}")

        for index, (key, value) in enumerate(sorted(nested)):
            if lines and lines[-1] != "":
                lines.append("")
            emit(prefix + (key,), value)

    emit((), data)
    if lines and lines[-1] != "":
        lines.append("")
    return "\n".join(lines)


@given(payload=nested_structured_mappings())
@PROPERTY_TEST_SETTINGS
def test_json_dumps_roundtrip(payload: dict[str, Any]) -> None:
    dumped = json_dumps(payload, ensure_ascii=False, default=list)
    parsed = json.loads(dumped)
    _assert_equivalent(parsed, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json_dumps(payload, ensure_ascii=False, default=list)
    safe_write(path, lambda handle: handle.write(text))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml

        # Convert tuples to lists for YAML compatibility
        def convert_tuples(obj):
            if isinstance(obj, dict):
                return {k: convert_tuples(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, deque)):
                return [convert_tuples(item) for item in obj]
            elif isinstance(obj, set):
                return [
                    convert_tuples(item)
                    for item in sorted(obj, key=lambda x: (type(x).__name__, repr(x)))
                ]
            return obj

        converted = convert_tuples(payload)
        text = yaml.dump(converted, default_flow_style=False)
    except ImportError:
        # Fallback to JSON if yaml not available
        text = json_dumps(payload, ensure_ascii=False, default=list)
    safe_write(path, lambda handle: handle.write(text))


def _write_toml(path: Path, payload: dict[str, Any]) -> None:
    text = _toml_dumps(_prepare_for_toml(payload))
    safe_write(path, lambda handle: handle.write(text))


@pytest.mark.parametrize(
    ("suffix", "writer"),
    (
        (".json", _write_json),
        (".yaml", _write_yaml),
        (".toml", _write_toml),
    ),
)
@given(payload=nested_structured_mappings())
@FILE_IO_PROPERTY_SETTINGS
def test_structured_file_roundtrip(
    tmp_path: Path, suffix: str, writer: Any, payload: dict[str, Any]
) -> None:
    destination = tmp_path / f"payload{suffix}"
    writer(destination, payload)
    loaded = read_structured_file(destination)
    _assert_equivalent(loaded, payload)


_MALFORMED_CASES = st.sampled_from(
    (
        (".json", "{", "Error parsing JSON file"),
        (".yaml", "key: [1", "Error parsing YAML file"),
        (".toml", "broken = [1,", "Error parsing TOML file"),
    )
)


@given(case=_MALFORMED_CASES)
@FILE_IO_PROPERTY_SETTINGS
def test_structured_file_error_contract(tmp_path: Path, case: tuple[str, str, str]) -> None:
    suffix, contents, expected_message = case
    path = tmp_path / f"invalid{suffix}"
    safe_write(path, lambda handle: handle.write(contents))

    with pytest.raises(StructuredFileError) as excinfo:
        read_structured_file(path)

    assert expected_message in str(excinfo.value)
