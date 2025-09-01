from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

from ..helpers import ensure_history
from ..sense import GLYPHS_CANONICAL
from .core import glifogram_series


def _write_csv(path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def export_history(G, base_path: str, fmt: str = "csv") -> None:
    """Vuelca glifograma y traza σ(t) a archivos CSV o JSON compactos."""
    hist = ensure_history(G)
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
    glifo = glifogram_series(G)
    sigma_mag = hist.get("sense_sigma_mag", [])
    sigma = {
        "t": list(range(len(sigma_mag))),
        "sigma_x": hist.get("sense_sigma_x", []),
        "sigma_y": hist.get("sense_sigma_y", []),
        "mag": sigma_mag,
        "angle": hist.get("sense_sigma_angle", []),
    }
    morph = hist.get("morph", [])
    epi_supp = hist.get("EPI_support", [])
    fmt = fmt.lower()
    if fmt == "csv":
        ts = glifo.get("t", [])
        default_col = [0] * len(ts)
        glif_rows = [
            [t] + [glifo.get(g, default_col)[i] for g in GLYPHS_CANONICAL]
            for i, t in enumerate(ts)
        ]
        _write_csv(base_path + "_glifogram.csv", ["t", *GLYPHS_CANONICAL], glif_rows)

        sigma_rows = [
            [t, sigma["sigma_x"][i], sigma["sigma_y"][i], sigma["mag"][i], sigma["angle"][i]]
            for i, t in enumerate(sigma["t"])
        ]
        _write_csv(base_path + "_sigma.csv", ["t", "x", "y", "mag", "angle"], sigma_rows)

        if morph:
            morph_rows = [
                [row.get("t"), row.get("ID"), row.get("CM"), row.get("NE"), row.get("PP")]
                for row in morph
            ]
            _write_csv(base_path + "_morph.csv", ["t", "ID", "CM", "NE", "PP"], morph_rows)
        if epi_supp:
            epi_rows = [
                [row.get("t"), row.get("size"), row.get("epi_norm")]
                for row in epi_supp
            ]
            _write_csv(base_path + "_epi_support.csv", ["t", "size", "epi_norm"], epi_rows)
    else:
        data = {"glifogram": glifo, "sigma": sigma, "morph": morph, "epi_support": epi_supp}
        with open(base_path + ".json", "w", encoding="utf-8") as f:
            json.dump(data, f)
