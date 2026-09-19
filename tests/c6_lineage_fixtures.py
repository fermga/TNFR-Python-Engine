"""Portable byte-lineage inputs, deliberately not scientific C6 evidence."""

import hashlib
import json
from copy import deepcopy


def write_synthetic_c6_lineage(directory, count):
    """Write a minimal B26-style input DAG for CLI prevalidation only.

    Records contain matching producer metadata and hashes, but no nodal states
    or admitted experiment manifest. Tests must stop at the analysis boundary;
    accepting these byte links does not admit a scientific report.
    """
    if not 2 <= count <= 6:
        raise ValueError("the byte-lineage fixture supports two through six inputs")
    key_rows = (
        (),
        ("input_evidence",),
        ("input_evidence", "source_input_evidence"),
        ("input_evidence", "parent_input_evidence", "source_input_evidence"),
        (
            "input_evidence",
            "parent_input_evidence",
            "previous_input_evidence",
            "source_input_evidence",
        ),
        (
            "input_evidence",
            "parent_input_evidence",
            "previous_input_evidence",
            "earlier_input_evidence",
            "source_input_evidence",
        ),
    )
    paths = tuple(directory / f"source_{index}.json" for index in range(count))
    records = [None] * count
    raw = [None] * count
    for index in reversed(range(count)):
        record = {
            "manifest": {"claim_id": f"synthetic-byte-source-{index}"},
            "source_scope": ["synthetic/byte-prevalidation-only"],
        }
        for key, target in zip(
            key_rows[count - index - 1], range(index + 1, count), strict=True
        ):
            record[key] = {
                "sha256": hashlib.sha256(raw[target]).hexdigest(),
                "producer_manifest": deepcopy(records[target]["manifest"]),
                "producer_source_scope": list(records[target]["source_scope"]),
            }
        records[index] = record
        raw[index] = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        paths[index].write_bytes(raw[index])
    return paths
