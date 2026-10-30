"""Constructed constant-field archive fixtures, never retained TNFR evidence.

Every derivative and whole-time statement here refers to the explicitly
manufactured constant vector field. The sine-shaped graph metadata tests
association only; it does not identify that artificial field with sine flow.
No source loader, differential field, jet generator or solver is called.
"""

from dataclasses import asdict
from fractions import Fraction as Q
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import networkx as nx

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.research.artifact_io import encode_exact_tree, file_receipt, sha256_bytes
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.maintained_sha_cut_separator import _capture_cut_source
from tnfr.research.retained_graph_state import record_graph_state
from tnfr.utils.io import json_dumps

STEP = Q(1, 512)
COUNT = 256
ORDER = 12
MARGIN = Q(1, 64)
MAX_RECORD_BYTES = 1024**2
MAX_ZIP_BYTES = 256 * 1024
PRODUCER_SCOPE = (
    "Manufactured constant-field affine solutions with exact dyadic carry. "
    "All higher derivatives and remainders are zero under that supplied field. "
    "The graph/law-shaped metadata exercises passive source association only; "
    "it does not prove that these records solve the held sine law. No actual "
    "TNFR source, retained record, field, jet or solver is consumed."
)


def _box(values):
    return tuple({"lo": value.lo, "hi": value.hi} for value in values)


def _json_bytes(value):
    return (json_dumps(encode_exact_tree(value), allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _exclusive(path, raw, *, expanded_bytes, record_kind):
    del expanded_bytes, record_kind
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def _graph():
    graph = nx.Graph()
    nodes = tuple(range(20)) + ("2_sub_0",)
    for index, node in enumerate(nodes):
        graph.add_node(
            node,
            EPI=Q(index - 4, 32),
            theta=Q(7) + Q(index, 64),
            nu_f=Q(4 + index % 5, 8),
            delta_nfr=Q(0),
        )
    edges = {
        tuple(sorted((i, (i + distance) % 20)))
        for distance in (1, 2)
        for i in range(20)
    }
    edges.update(tuple(sorted((i, (i + 3) % 20))) for i in range(12))
    for index, (left, right) in enumerate(sorted(edges)):
        graph.add_edge(left, right, weight=Q(3 + index % 7, 4))
    graph.add_edge(2, "2_sub_0", weight=Q(3, 2))
    graph.graph.update(_t=Q(7) + Q(1, 2**19), GAMMA={"type": "none"})
    return graph


def constant_step(initial, *, branch, index):
    """Derive a record directly from x(t)=x(0)+rate*t, without Taylor code."""
    if branch not in ("connected", "cut_host"):
        raise ValueError("constructed branch is not declared")
    expected = 42 if branch == "connected" else 40
    if len(initial) != expected or type(index) is not int or not 0 <= index < COUNT:
        raise ValueError("constructed step shape/index differs")
    rates = tuple(I(Q(1 if i % 2 == 0 else -1, 16)) for i in range(expected))
    increment = tuple(I(rate.lo * STEP) for rate in rates)
    endpoint = tuple(
        I(value.lo + change.lo, value.hi + change.hi)
        for value, change in zip(initial, increment, strict=True)
    )
    tube = tuple(
        I(
            value.lo + min(Q(0), change.lo) - MARGIN,
            value.hi + max(Q(0), change.hi) + MARGIN,
        )
        for value, change in zip(initial, increment, strict=True)
    )
    step = {
        "time": index * STEP,
        "duration": STEP,
        "order": ORDER,
        "initial_box": _box(initial),
        "tube": _box(tube),
        "series": tuple(
            _box((value, rate) + (I(0),) * (ORDER - 1))
            for value, rate in zip(initial, rates, strict=True)
        ),
        "local_remainder_bounds": _box((I(0),) * expected),
        "increment": _box(increment),
        "endpoint": _box(endpoint),
        "picard_interior_margin": MARGIN,
        "domain_lower_bounds": (Q(1),),
        "method": "direct_source_box_Picard_Taylor_dyadic128_v1",
    }
    record = {"index": index, "time": index * STEP, "step": step}
    if branch == "connected":
        # Retained as inherited producer metadata, not recomputed sine margins.
        record.update(
            tube_principal_margins=(Q(1),) * 53, combined_principal_margins=(Q(1),) * 53
        )
    else:
        record.update(
            failed_tube=None,
            verification={
                "tube_rates": _box(rates),
                "picard_interior_margin": MARGIN,
                "solution_order": ORDER,
                "remainder_order": ORDER + 1,
                "series_verified": True,
                "remainder_verified": True,
                "arithmetic_verified": True,
            },
            observation={},
            accepted=True,
            reason=None,
        )
    return record, endpoint, rates


def encoded_step(record, *, padding_bytes=0):
    """Preserve exact JSON values while optionally padding raw byte throughput."""
    if type(padding_bytes) is not int or not 0 <= padding_bytes <= MAX_RECORD_BYTES:
        raise ValueError("constructed padding exceeds the declared member budget")
    raw = _json_bytes(record)
    if padding_bytes:
        if len(raw) > padding_bytes:
            raise ValueError("constructed record exceeds its fixed padding target")
        raw += b" " * (padding_bytes - len(raw))
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("constructed member exceeds byte budget")
    member = f"step-{record['index']:03d}.json"
    buffer = BytesIO()
    info = ZipInfo(member, date_time=(2026, 10, 23, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    with ZipFile(buffer, "w") as archive:
        archive.writestr(info, raw)
    compressed = buffer.getvalue()
    if len(compressed) > MAX_ZIP_BYTES:
        raise ValueError("constructed ZIP exceeds byte budget")
    return raw, compressed, member


def build_phase_fixture(
    folder, *, branch, padding_bytes=0, commit=None, checkpoint=None
):
    """Write one complete bounded 256-cell branch and public loader arguments.

    ``commit(path, raw, expanded_bytes=..., record_kind=...)`` may enforce an
    external cumulative budget. The default writes exclusively. ``checkpoint``
    allows an outer resource deadline, without changing any scientific input.
    All returned receipts are relative to this new constructed folder.
    """
    if branch not in ("connected", "cut_host"):
        raise ValueError("constructed branch is not declared")
    if type(padding_bytes) is not int or not 0 <= padding_bytes <= MAX_RECORD_BYTES:
        raise ValueError("constructed padding exceeds member budget")
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError("constructed fixture folder must be empty")
    commit = _exclusive if commit is None else commit
    checkpoint = (lambda: None) if checkpoint is None else checkpoint
    inputs = []

    def save(name, payload):
        checkpoint()
        raw = _json_bytes(payload)
        if len(raw) > MAX_RECORD_BYTES:
            raise ValueError("constructed metadata exceeds byte budget")
        path = folder / name
        commit(path, raw, expanded_bytes=len(raw), record_kind="metadata")
        item = file_receipt(path, label=name)
        inputs.append(item)
        return item

    graph = _graph()
    captured = _capture_cut_source(
        graph,
        parent=2,
        child="2_sub_0",
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_domain="regular"
        ),
    )
    primitive_receipt = save(
        "constructed-primitives.json",
        {
            "schema": "tnfr.constructed-primitive-association.v1",
            "graph": record_graph_state(graph),
            "scope": PRODUCER_SCOPE,
        },
    )
    source_receipts = (primitive_receipt,)
    source = asdict(captured)
    binding = {
        "complete_source": source,
        "source_receipts": source_receipts,
        "source_graph_signature": _signature(graph),
    }
    indices = tuple(range(21)) if branch == "connected" else captured.host_indices
    node_order = tuple(captured.nodes[i] for i in indices)
    initial = tuple(I(captured.epi[i]) for i in indices) + tuple(
        I(captured.phase[i]) for i in indices
    )
    loader = {
        "branch": branch,
        "node_order": node_order,
        "initial_box": initial,
        "source_binding": binding,
    }
    if branch == "connected":
        source_receipt = save(
            "source.json",
            {
                "schema": "tnfr.maintained-sha-retention-source.v1",
                "endpoint": record_graph_state(graph),
                "retained_source_receipts": source_receipts,
                "initial_box": _box(initial),
            },
        )
        protocol = save(
            "protocol.json",
            {
                "schema": "tnfr.maintained-sha-retention-protocol.v1",
                "node_order": node_order,
                "law": "x_dot=M*S(theta); theta_dot=M*L_W*x; M=diag(nu/(pi*d))",
                "prior_clock": captured.structural_time,
                "horizon": Q(1, 2),
                "step": STEP,
                "maximum_steps": COUNT,
                "order": ORDER,
            },
        )
        freeze = save(
            "freeze.json",
            {
                "schema": "tnfr.maintained-sha-retention-freeze.v1",
                "artifacts": (source_receipt, protocol),
            },
        )
        attempt = save(
            "attempt.json",
            {
                "schema": "tnfr.maintained-sha-retention-attempt.v1",
                "freeze": freeze,
            },
        )
        producers = (protocol, attempt, freeze)
        loader["source_receipt"] = source_receipt
    else:
        protocol = save(
            "protocol.json", {"schema": "constructed", "scope": PRODUCER_SCOPE}
        )
        attempt = save(
            "attempt.json",
            {
                "schema": "tnfr.cut-direct-response-attempt.v1",
                "protocol": protocol,
                "reference_clock": captured.structural_time,
                "window": Q(1, 2),
            },
        )
        producers = (protocol, attempt)
    records, observations = [], []
    box, expanded_total, archive_total = initial, 0, 0
    for index in range(COUNT):
        checkpoint()
        record, endpoint, rates = constant_step(box, branch=branch, index=index)
        raw, compressed, member = encoded_step(record, padding_bytes=padding_bytes)
        expanded_total += len(raw)
        archive_total += len(compressed)
        if expanded_total > 128 * 1024**2 or archive_total > 64 * 1024**2:
            raise ValueError("constructed branch archive budget exceeded")
        name = f"steps/{index:03d}.zip"
        commit(folder / name, compressed, expanded_bytes=len(raw), record_kind="step")
        receipt = file_receipt(folder / name, label=name)
        inputs.append(receipt)
        if branch == "connected":
            records.append(
                {
                    "artifact": receipt,
                    "member": {
                        "path": member,
                        "bytes": len(raw),
                        "sha256": sha256_bytes(raw),
                    },
                }
            )
        else:
            records.append(
                {
                    "index": index,
                    "time": index * STEP,
                    "accepted": True,
                    "archive": receipt,
                    "member": member,
                    "expanded_bytes": len(raw),
                    "expanded_sha256": sha256_bytes(raw),
                }
            )
            observations.append(record["observation"])
        box = endpoint
        checkpoint()
    if branch == "connected":
        result = {
            "schema": "tnfr.maintained-sha-retention-result.v1",
            "attempt": attempt,
            "report": {
                "status": "completed",
                "attempted_steps": COUNT,
                "accepted_steps": COUNT,
                "certified_retention_time": Q(1, 2),
                "full_horizon_retention": True,
            },
            "records": tuple(records),
        }
    else:
        result = {
            "schema": "tnfr.cut-direct-response-result.v1",
            "attempt": attempt,
            "protocol": protocol,
            "source_receipts": source_receipts,
            "report": {
                "preparation": {
                    "source": source,
                    "initial": _box(initial),
                    "source_receipts": source_receipts,
                    "persistent_source_unchanged": True,
                },
                "status": "completed",
                "reason": None,
                "attempted_steps": COUNT,
                "accepted_steps": COUNT,
                "certified_time": Q(1, 2),
                "cells": tuple(observations),
            },
            "step_records": tuple(records),
        }
    loader["result_receipt"] = save("result.json", result)
    loader["producer_receipts"] = producers
    return {
        "evidence_dir": folder,
        "loader_kwargs": loader,
        "rates": rates,
        "expected_endpoint": box,
        "input_receipts": tuple(inputs),
        "producer_scope": PRODUCER_SCOPE,
        "expanded_step_bytes": expanded_total,
        "stored_step_bytes": archive_total,
    }
