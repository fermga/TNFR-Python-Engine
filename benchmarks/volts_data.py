"""Bounded, identity-pinned ingestion of the Stat2Data Volts trace.

This optional research adapter does not execute R or admit a physical model.
The single acquisition is preserved; splitting it does not create independent
experimental runs. No download or analysis happens at import time.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import warnings
import zlib

from tnfr.validation.nodal_prediction import NodalMeasurementRun


SOURCE_SHA256 = "e50e505783dbf9b6d477165c92328e805e5d8bb0d27e3458862e2d5ba7f00a52"
SOURCE_BYTES = 767
MAX_COMPRESSED_BYTES = 1024
MAX_EXPANDED_BYTES = 65536
RDATA_VERSION = "1.1.0"
RUN_ID = "Stat2Data-Volts-single-trace"


def _rdata_module():
    try:
        module = importlib.import_module("rdata")
    except ImportError as exc:
        raise ImportError(
            "Volts ingestion requires the optional research dependency rdata==1.1.0"
        ) from exc
    if module.__version__ != RDATA_VERSION:
        raise ImportError(
            "Volts ingestion requires rdata==1.1.0 for reproducible decoding"
        )
    return module


def _bounded_expand(raw: bytes) -> bytes:
    """Expand exactly one complete gzip/zlib stream, never nested compression."""
    if not raw or len(raw) > MAX_COMPRESSED_BYTES:
        raise ValueError("compressed input exceeds the Volts byte bound")
    decoder = zlib.decompressobj(wbits=47)
    try:
        expanded = decoder.decompress(raw, MAX_EXPANDED_BYTES + 1)
    except zlib.error as exc:
        raise ValueError("Volts input must be a valid gzip/zlib stream") from exc
    if len(expanded) > MAX_EXPANDED_BYTES or decoder.unconsumed_tail:
        raise ValueError("expanded input exceeds the Volts byte bound")
    if not decoder.eof or decoder.unused_data:
        raise ValueError(
            "Volts compression must be complete with no trailing members or bytes"
        )
    if not expanded.startswith((b"RDX2\nX\n", b"RDX3\nX\n")):
        raise ValueError(
            "expected uncompressed XDR RData; nested compression is forbidden"
        )
    return expanded


def _validate_plain_tree(parsed, module) -> None:
    """Reject executable/custom R types before invoking any class conversion."""
    types = module.parser.RObjectType
    allowed = {
        types.NILVALUE,
        types.SYM,
        types.LIST,
        types.CHAR,
        types.INT,
        types.REAL,
        types.STR,
        types.VEC,
        types.REF,
    }
    pending = [(parsed.object, 0)]
    seen = set()
    while pending:
        obj, depth = pending.pop()
        if depth > 32 or len(seen) >= 512:
            raise ValueError("R object structure exceeds the Volts bound")
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        kind = obj.info.type
        if kind not in allowed or (obj.info.object and kind != types.VEC):
            raise ValueError("only plain data.frame and numeric R objects are accepted")
        if kind == types.REF:
            target = obj.referenced_object
            if target is None or target.info.type != types.SYM:
                raise ValueError(
                    "only previously defined R symbol references are accepted"
                )
            pending.append((target, depth + 1))
        for child in (obj.attributes, obj.tag):
            if child is not None:
                pending.append((child, depth + 1))
        if kind in {types.LIST, types.STR, types.VEC}:
            if len(obj.value) > 64:
                raise ValueError("R vector length exceeds the Volts bound")
            pending.extend((child, depth + 1) for child in obj.value)
        elif kind == types.SYM:
            pending.append((obj.value, depth + 1))
        elif kind in {types.INT, types.REAL}:
            if obj.value.ndim != 1 or len(obj.value) > 50:
                raise ValueError("R numeric vector exceeds the Volts row bound")
            if obj.attributes is not None:
                raise ValueError("Volts numeric columns must have no custom attributes")
        elif kind == types.CHAR:
            if obj.value is None or len(obj.value) > 128:
                raise ValueError("invalid or oversized R metadata string")
            try:
                obj.value.decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    "Volts metadata must contain only ASCII characters"
                ) from exc


def _validated_run(converted) -> NodalMeasurementRun:
    import numpy as np
    import pandas as pd

    if type(converted) is not dict or tuple(converted) != ("Volts",):
        raise ValueError("RData must contain exactly one dataset named Volts")
    frame = converted["Volts"]
    if type(frame) is not pd.DataFrame or frame.shape != (50, 2):
        raise ValueError(
            "Volts must be a plain data.frame with exactly 50 rows and two columns"
        )
    if tuple(frame.columns) != ("Voltage", "Time"):
        raise ValueError("Volts columns must be exactly Voltage, Time in that order")
    for column in frame.columns:
        if frame[column].dtype.kind not in "fiu":
            raise ValueError(
                "Volts columns must be real numeric, not boolean or textual"
            )
    values = frame.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError("Volts measurements and timestamps must all be finite")
    return NodalMeasurementRun(
        run_id=RUN_ID,
        acquisition_id=RUN_ID,
        channel_ids=("Voltage",),
        timestamps=tuple(values[:, 1]),
        samples=(tuple(values[:, 0]),),
        value_unit="V",
        time_unit="s",
    )


def _decode_expanded(expanded: bytes) -> tuple[NodalMeasurementRun, dict]:
    """Internal synthetic-test seam; public ingestion first authenticates bytes."""
    if len(expanded) > MAX_EXPANDED_BYTES:
        raise ValueError("expanded input exceeds the Volts byte bound")
    if not expanded.startswith((b"RDX2\nX\n", b"RDX3\nX\n")):
        raise ValueError(
            "expected uncompressed XDR RData; nested compression is forbidden"
        )
    module = _rdata_module()
    parsed = module.parser.parse_data(
        expanded,
        expand_altrep=False,
        altrep_constructor_dict={},
        extension=".rda",
    )
    _validate_plain_tree(parsed, module)

    def plain_frame(obj, attrs):
        if set(attrs) != {"names", "class", "row.names"} or tuple(attrs["class"]) != (
            "data.frame",
        ):
            raise ValueError(
                "only the plain data.frame class and metadata are accepted"
            )
        return module.conversion.dataframe_constructor(obj, attrs)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converted = module.conversion.convert(
            parsed,
            constructor_dict={"data.frame": plain_frame},
            default_encoding="ascii",
        )
    run = _validated_run(converted)
    return run, {
        "expanded_bytes": len(expanded),
        "expanded_sha256": hashlib.sha256(expanded).hexdigest(),
        "rdata_version": module.__version__,
        "serialization_format": parsed.versions.format,
    }


def load_volts(path: str | Path) -> tuple[NodalMeasurementRun, dict]:
    """Load only the preregistered archive; preserve every original observation."""
    with Path(path).open("rb") as source:
        raw = source.read(MAX_COMPRESSED_BYTES + 1)
    if len(raw) > MAX_COMPRESSED_BYTES:
        raise ValueError("compressed input exceeds the Volts byte bound")
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != SOURCE_BYTES or digest != SOURCE_SHA256:
        raise ValueError(
            "Volts source size or SHA-256 differs from the preregistered archive"
        )
    expanded = _bounded_expand(raw)
    run, metadata = _decode_expanded(expanded)
    metadata.update(
        {
            "source_bytes": len(raw),
            "source_sha256": digest,
            "dataset": "Volts",
            "rows": 50,
            "columns": ("Voltage", "Time"),
            "acquisition_id": RUN_ID,
            "split_scope": "within_single_acquisition",
            "physical_status": "not_admitted",
        }
    )
    return run, metadata
