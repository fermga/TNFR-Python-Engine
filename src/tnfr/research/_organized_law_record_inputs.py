"""Read the fixed scalar-moment archive without replaying its producer.

Byte receipts bind this calculation to retained evidence. They authenticate
neither a physical observation nor the truth of an integral enclosure.
The consuming theorem re-admits primitive moments and rebuilds all contrasts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..mathematics._rational_interval import I
from ..physics._organized_law_contrast_kernels import _MomentKernel
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    exact_record,
    read_bytes_bounded,
    sha256_bytes,
)

_REVISION = "0b21a040270c7c7653cd09e9f3076fdb1ca9e08f"
_HASHES = {
    "attempt.json": "da33f61594c40ae1ce2a0d046086d103c433c267015c7cbbda2cdffe9181977e",
    "result.json": "b8412aca8089a185466d96bf29e89aeccbc5d3c05117dcc47f3b537bb2e7fc4f",
}


def _read_organized_law_record_moments(
    repository: Path,
) -> tuple[tuple[_MomentKernel, ...], tuple[dict[str, Any], ...]]:
    """Return the five preselected mode-two primitives and exact byte receipts."""
    directory = Path(repository) / "docs/assets/organized_law_contrast_kernels"
    records, receipts = {}, []
    for name, expected in _HASHES.items():
        raw = read_bytes_bounded(directory / name, max_bytes=2_000_000)
        digest = sha256_bytes(raw)
        if digest != expected:
            raise ValueError("fixed moment archive byte association has changed")
        records[name] = decode_exact_tree(json_loads(raw.decode("utf-8")))
        receipts.append({"path": name, "bytes": len(raw), "sha256": digest})
    attempt, result = records["attempt.json"], records["result.json"]
    if (
        attempt["source_revision"] != _REVISION
        or result["source_revision"] != _REVISION
        or result["attempt"] != receipts[0]
        or attempt["schema"] != "organized_law_contrast_kernels_attempt_v1"
        or result["schema"] != "organized_law_contrast_kernels_result_v1"
    ):
        raise ValueError("fixed moment archive source association is inconsistent")

    def interval(value: Any) -> I:
        if not isinstance(value, dict) or set(value) != {"lo", "hi"}:
            raise ValueError("moment interval requires explicit endpoints")
        return I(exact_record(value["lo"]), exact_record(value["hi"]))

    selected = []
    for item in result["report"]["kernels"]:
        if type(item["mode"]) is not int or type(item["harmonic"]) is not int:
            raise ValueError("moment labels require ordinary integers")
        if item["mode"] == 2:
            selected.append(
                _MomentKernel(
                    mode=item["mode"],
                    harmonic=item["harmonic"],
                    mu=interval(item["mu"]),
                    sine_hyperbolic=interval(item["sine_hyperbolic"]),
                    cosine_hyperbolic=interval(item["cosine_hyperbolic"]),
                    sine_truncation=exact_record(item["sine_truncation"]),
                    cosine_truncation=exact_record(item["cosine_truncation"]),
                )
            )
    if tuple(item.harmonic for item in selected) != (1, 2, 9, 18, 36):
        raise ValueError("fixed mode-two moment inventory is incomplete")
    return tuple(selected), tuple(receipts)
