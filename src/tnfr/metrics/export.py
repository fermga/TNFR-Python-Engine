"""Metrics export."""

from __future__ import annotations

import csv
from itertools import zip_longest, tee

from ..glyph_history import ensure_history
from ..io import safe_write
from ..constants_glyphs import GLYPHS_CANONICAL
from .core import glyphogram_series
from ..json_utils import json_dumps


def _write_csv(path, headers, rows):
    def _write(f):
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)

    safe_write(path, _write, newline="")


def _iter_glif_rows(glyph):
    ts = glyph.get("t", [])
    # Precompute columns for each glyph to avoid repeated lookups.
    # ``default_col`` is shared by reference for missing glyphs to prevent
    # unnecessary list allocations.
    default_col = [0] * len(ts)
    cols = [glyph.get(g, default_col) for g in GLYPHS_CANONICAL]
    for i, t in enumerate(ts):
        yield [t] + [col[i] for col in cols]


def export_metrics(G, base_path: str, fmt: str = "csv") -> None:
    """Dump glyphogram and σ(t) trace to compact CSV or JSON files."""
    hist = ensure_history(G)
    glyph = glyphogram_series(G)
    sigma_x = hist.get("sense_sigma_x", [])
    sigma_y = hist.get("sense_sigma_y", [])
    sigma_mag = hist.get("sense_sigma_mag", [])
    sigma_angle = hist.get("sense_sigma_angle", [])
    t_series = hist.get("sense_sigma_t", []) or glyph.get("t", [])
    rows_raw = zip_longest(
        t_series, sigma_x, sigma_y, sigma_mag, sigma_angle, fillvalue=None
    )

    def _gen_rows():
        for i, (t, x, y, m, a) in enumerate(rows_raw):
            yield (
                i if t is None else t,
                x or 0,
                y or 0,
                m or 0,
                a or 0,
            )

    rows_csv, rows_sigma = tee(_gen_rows())

    sigma = {"t": [], "sigma_x": [], "sigma_y": [], "mag": [], "angle": []}
    for t, x, y, m, a in rows_sigma:
        sigma["t"].append(t)
        sigma["sigma_x"].append(x)
        sigma["sigma_y"].append(y)
        sigma["mag"].append(m)
        sigma["angle"].append(a)
    morph = hist.get("morph", [])
    epi_supp = hist.get("EPI_support", [])
    fmt = fmt.lower()
    if fmt not in {"csv", "json"}:
        raise ValueError(f"Formato de exportación no soportado: {fmt}")
    if fmt == "csv":
        specs = [
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
                    (
                        [row.get("t"), row.get("size"), row.get("epi_norm")]
                        for row in epi_supp
                    ),
                )
            )
        for suffix, headers, rows in specs:
            _write_csv(base_path + suffix, headers, rows)
    else:
        data = {
            "glyphogram": glyph,
            "sigma": sigma,
            "morph": morph,
            "epi_support": epi_supp,
        }
        json_path = base_path + ".json"
        safe_write(
            json_path,
            lambda f: f.write(json_dumps(data).decode("utf-8")),
        )
