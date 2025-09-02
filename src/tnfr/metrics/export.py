"""Exportación de métricas."""
from __future__ import annotations

import csv
import json

from ..helpers import ensure_history, ensure_parent
from ..constants_glifos import GLYPHS_CANONICAL
from .core import glifogram_series


def _write_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def _iter_glif_rows(glifo):
    ts = glifo.get("t", [])
    default_col = [0] * len(ts)
    for i, t in enumerate(ts):
        yield [t] + [glifo.get(g, default_col)[i] for g in GLYPHS_CANONICAL]


def _iter_sigma_rows(sigma):
    return (
        [t, x, y, m, a]
        for t, x, y, m, a in zip(
            sigma["t"],
            sigma["sigma_x"],
            sigma["sigma_y"],
            sigma["mag"],
            sigma["angle"],
        )
    )


def export_history(G, base_path: str, fmt: str = "csv") -> None:
    """Vuelca glifograma y traza σ(t) a archivos CSV o JSON compactos."""
    hist = ensure_history(G)
    ensure_parent(base_path)
    glifo = glifogram_series(G)
    sigma_x = hist.tracked_get("sense_sigma_x", [])
    sigma_y = hist.tracked_get("sense_sigma_y", [])
    sigma_mag = hist.tracked_get("sense_sigma_mag", [])
    sigma_angle = hist.tracked_get("sense_sigma_angle", [])
    min_len = min(len(sigma_x), len(sigma_y), len(sigma_mag), len(sigma_angle))
    sigma = {
        "t": list(range(min_len)),
        "sigma_x": sigma_x[:min_len],
        "sigma_y": sigma_y[:min_len],
        "mag": sigma_mag[:min_len],
        "angle": sigma_angle[:min_len],
    }
    morph = hist.tracked_get("morph", [])
    epi_supp = hist.tracked_get("EPI_support", [])
    fmt = fmt.lower()
    if fmt == "csv":
        specs = [
            ("_glifogram.csv", ["t", *GLYPHS_CANONICAL], _iter_glif_rows(glifo)),
            ("_sigma.csv", ["t", "x", "y", "mag", "angle"], _iter_sigma_rows(sigma)),
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
                        [row.get("t"), row.get("size"), row.get("epi_norm")] for row in epi_supp
                    ),
                )
            )
        for suffix, headers, rows in specs:
            _write_csv(base_path + suffix, headers, rows)
    else:
        data = {"glifogram": glifo, "sigma": sigma, "morph": morph, "epi_support": epi_supp}
        with open(base_path + ".json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
