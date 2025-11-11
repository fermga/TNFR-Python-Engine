"""Metrics export."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Iterator, Sequence
from itertools import tee, zip_longest
from typing import Mapping, TextIO

from ..config.constants import GLYPHS_CANONICAL
from ..glyph_history import ensure_history
from ..utils import json_dumps, safe_write
from ..types import Graph, SigmaTrace
from .core import glyphogram_series


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


def _iter_glif_rows(
    glyph: Mapping[str, Sequence[float]],
) -> Iterator[list[float]]:
    ts = glyph.get("t", [])
    # Precompute columns for each glyph to avoid repeated lookups.
    # ``default_col`` is shared by reference for missing glyphs to prevent
    # unnecessary list allocations.
    default_col = [0] * len(ts)
    cols = [glyph.get(g, default_col) for g in GLYPHS_CANONICAL]
    for i, t in enumerate(ts):
        yield [t] + [col[i] for col in cols]


def export_metrics(
    G: Graph,
    base_path: str,
    fmt: str = "csv",
    *,
    output_dir: str | None = None,
) -> None:
    """Dump glyphogram and σ(t) trace to compact CSV or JSON files.

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

    hist = ensure_history(G)
    glyph = glyphogram_series(G)
    sigma_x = hist.get("sense_sigma_x", [])
    sigma_y = hist.get("sense_sigma_y", [])
    sigma_mag = hist.get("sense_sigma_mag", [])
    sigma_angle = hist.get("sense_sigma_angle", [])
    t_series = hist.get("sense_sigma_t", []) or glyph.get("t", [])
    rows_raw = zip_longest(t_series, sigma_x, sigma_y, sigma_mag, sigma_angle, fillvalue=None)

    def _clean(value: float | None) -> float:
        """Return ``0`` for ``None`` or ``NaN`` values."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0
        return value

    def _gen_rows() -> Iterator[tuple[float, float, float, float, float]]:
        for i, (t, x, y, m, a) in enumerate(rows_raw):
            yield (
                i if t is None else t,
                _clean(x),
                _clean(y),
                _clean(m),
                _clean(a),
            )

    rows_csv, rows_sigma = tee(_gen_rows())

    sigma: SigmaTrace = {
        "t": [],
        "sigma_x": [],
        "sigma_y": [],
        "mag": [],
        "angle": [],
    }
    for t, x, y, m, a in rows_sigma:
        sigma["t"].append(t)
        sigma["sigma_x"].append(x)
        sigma["sigma_y"].append(y)
        sigma["mag"].append(m)
        sigma["angle"].append(a)
    morph: Sequence[Mapping[str, float]] = hist.get("morph", [])
    epi_supp: Sequence[Mapping[str, float]] = hist.get("EPI_support", [])
    fmt = fmt.lower()
    if fmt not in {"csv", "json"}:
        raise ValueError(f"Unsupported export format: {fmt}")
    if fmt == "csv":
        specs: list[tuple[str, Sequence[str], Iterable[Sequence[object]]]] = [
            (
                "_glyphogram.csv",
                ["t", *GLYPHS_CANONICAL],
                _iter_glif_rows(glyph),
            ),
            (
                "_sigma.csv",
                ["t", "x", "y", "mag", "angle"],
                ([t, x, y, m, a] for t, x, y, m, a in rows_csv),
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
                    ([row.get("t"), row.get("size"), row.get("epi_norm")] for row in epi_supp),
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
            f.write(json_dumps(data))

        safe_write(json_path, _write_json, base_dir=output_dir)
