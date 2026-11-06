"""Admitted stored-history snapshots shared by CSV and JSON exports."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TextIO, cast

from ..config.constants import GLYPHS_CANONICAL
from ..errors import TNFRValueError
from ..glyph_history import ensure_history as ensure_history  # noqa: F401
from ..types import Graph, SigmaTrace
from ..utils import json_dumps, safe_write
from ._history import stored_sequence as _stored_sequence
from ._history import stored_value as _stored_value
from .reporting import glyphogram_series as glyphogram_series  # noqa: F401


def _write_csv(
    path: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    output_dir: str | None = None,
) -> None:
    def _write(f: TextIO) -> None:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)

    safe_write(path, _write, newline="", base_dir=output_dir)


def _stored_rows(value: Any, label: str) -> list[dict[str, float | None]]:
    rows = []
    for index, row in enumerate(_stored_sequence(value, label)):
        if not isinstance(row, Mapping):
            raise TypeError(f"{label}[{index}] must be a mapping")
        if any(not isinstance(key, str) for key in row):
            raise TypeError(f"{label}[{index}] keys must be strings")
        rows.append(
            {
                key: _stored_value(value, f"{label}[{index}].{key}")
                for key, value in row.items()
            }
        )
    return rows


def export_metrics(
    G: Graph,
    base_path: str,
    fmt: str = "csv",
    *,
    output_dir: str | None = None,
) -> None:
    """Export one admitted snapshot of stored metric history.

    Missing samples and clocks remain ``null`` in JSON and empty CSV cells.
    Sigma columns align by retained position only; timestamps are never borrowed
    from another stream or inferred from row indices. Present nonfinite,
    Boolean, nonreal or unrepresentable values reject before any output is
    replaced. The graph and its history are not changed.

    Each output uses the shared atomic writer. The set of CSV files is not one
    transaction: filesystem failure can leave earlier files replaced.

    Parameters
    ----------
    G : Graph
        The TNFR graph containing metrics to export.
    base_path : str
        Base filename for exported files (without extension).
    fmt : str, default='csv'
        Export format: 'csv' or 'json'.
    output_dir : str | None, optional
        Output directory to restrict exports. If provided, all exports
        must stay within this directory (prevents path traversal).

    Raises
    ------
    ValueError
        If the path is invalid or format is unsupported.
    PathTraversalError
        If path traversal is detected when output_dir is provided.
    """

    if not isinstance(fmt, str):
        raise TypeError("fmt must be a string")
    fmt = fmt.lower()
    if fmt not in {"csv", "json"}:
        raise TNFRValueError(
            f"Unsupported export format: {fmt}",
            suggestion="Use 'csv' or 'json'",
            context={"format": fmt, "supported": ["csv", "json"]},
        )
    hist = G.graph.get("history", {})
    if not isinstance(hist, Mapping):
        raise TypeError("history must be a mapping")
    glyph_rows = _stored_rows(hist.get("glyphogram", ()), "glyphogram")
    glyph_headers = ("t", *GLYPHS_CANONICAL)
    glyph = (
        {key: [row.get(key) for row in glyph_rows] for key in glyph_headers}
        if glyph_rows
        else {"t": []}
    )
    sigma_sources = {
        "t": "sense_sigma_t",
        "sigma_x": "sense_sigma_x",
        "sigma_y": "sense_sigma_y",
        "mag": "sense_sigma_mag",
        "angle": "sense_sigma_angle",
    }
    sigma_columns: dict[str, list[float | None]] = {
        key: [
            _stored_value(value, f"{source}[{index}]")
            for index, value in enumerate(
                _stored_sequence(hist.get(source, ()), source)
            )
        ]
        for key, source in sigma_sources.items()
    }
    sample_count = max(map(len, sigma_columns.values()), default=0)
    for values in sigma_columns.values():
        values.extend([None] * (sample_count - len(values)))
    sigma = cast(SigmaTrace, sigma_columns)
    morph = _stored_rows(hist.get("morph", ()), "morph")
    epi_supp = _stored_rows(hist.get("EPI_support", ()), "EPI_support")
    if fmt == "csv":
        specs: list[tuple[str, Sequence[str], Iterable[Sequence[object]]]] = [
            (
                "_glyphogram.csv",
                glyph_headers,
                ([row.get(key) for key in glyph_headers] for row in glyph_rows),
            ),
            (
                "_sigma.csv",
                ["t", "x", "y", "mag", "angle"],
                zip(*(sigma_columns[key] for key in sigma_sources)),
            ),
        ]
        if morph:
            specs.append(
                (
                    "_morph.csv",
                    ["t", "ID", "CM", "NE", "PP"],
                    (
                        [
                            row.get("t"),
                            row.get("ID"),
                            row.get("CM"),
                            row.get("NE"),
                            row.get("PP"),
                        ]
                        for row in morph
                    ),
                )
            )
        if epi_supp:
            specs.append(
                (
                    "_epi_support.csv",
                    ["t", "size", "epi_norm"],
                    (
                        [row.get("t"), row.get("size"), row.get("epi_norm")]
                        for row in epi_supp
                    ),
                )
            )
        for suffix, headers, rows in specs:
            _write_csv(base_path + suffix, headers, rows, output_dir=output_dir)
    else:
        data = {
            "glyphogram": glyph,
            "sigma": sigma,
            "morph": morph,
            "epi_support": epi_supp,
        }
        json_path = base_path + ".json"

        def _write_json(f: TextIO) -> None:
            f.write(cast(str, json_dumps(data)))

        safe_write(json_path, _write_json, base_dir=output_dir)
