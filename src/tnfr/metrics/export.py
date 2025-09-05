"""Metrics export."""

from __future__ import annotations

import csv
import json
from itertools import zip_longest

from ..glyph_history import ensure_history
from ..helpers import ensure_parent
from ..constants_glyphs import GLYPHS_CANONICAL
from .core import glyphogram_series


def _write_csv(path, headers, rows):
    ensure_parent(path)
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)
    except OSError as e:
        raise OSError(f"Failed to write CSV file {path}: {e}") from e


def _iter_glif_rows(glyph):
    ts = glyph.get("t", [])
    default_col = [0] * len(ts)
    for i, t in enumerate(ts):
        yield [t] + [glyph.get(g, default_col)[i] for g in GLYPHS_CANONICAL]


def _iter_sigma_rows(sigma_rows):
    return ([t, x, y, m, a] for t, (x, y, m, a) in enumerate(sigma_rows))


def export_history(G, base_path: str, fmt: str = "csv") -> None:
    """Dump glyphogram and σ(t) trace to compact CSV or JSON files."""
    hist = ensure_history(G)
    ensure_parent(base_path)
    glyph = glyphogram_series(G)
    sigma_x = hist.tracked_get("sense_sigma_x", [])
    sigma_y = hist.tracked_get("sense_sigma_y", [])
    sigma_mag = hist.tracked_get("sense_sigma_mag", [])
    sigma_angle = hist.tracked_get("sense_sigma_angle", [])
    sigma_rows = list(
        zip_longest(sigma_x, sigma_y, sigma_mag, sigma_angle, fillvalue=0)
    )
    sigma = {
        "t": list(range(len(sigma_rows))),
        "sigma_x": [x for x, _, _, _ in sigma_rows],
        "sigma_y": [y for _, y, _, _ in sigma_rows],
        "mag": [m for _, _, m, _ in sigma_rows],
        "angle": [a for _, _, _, a in sigma_rows],
    }
    morph = hist.tracked_get("morph", [])
    epi_supp = hist.tracked_get("EPI_support", [])
    fmt = fmt.lower()
    if fmt == "csv":
        specs = [
            ("_glyphogram.csv", ["t", *GLYPHS_CANONICAL], _iter_glif_rows(glyph)),
            (
                "_sigma.csv",
                ["t", "x", "y", "mag", "angle"],
                _iter_sigma_rows(sigma_rows),
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
        ensure_parent(json_path)
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            raise OSError(f"Failed to write JSON file {json_path}: {e}") from e
