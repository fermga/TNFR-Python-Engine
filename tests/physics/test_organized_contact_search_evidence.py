"""Read-only reconstruction of frozen endpoints; never replay the search."""

from fractions import Fraction as Q
import hashlib
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.physics._organized_external_interaction import (
    _derive_organized_external_interaction,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/organized_contact_stationary/search-v1"
REVISION = "75497035c5e0840b164897119ce5a50dcf6bda91"


def _phases(values):
    if type(values) is not list or len(values) != 54:
        raise ValueError("expected 54 hexadecimal phase coordinates")
    result = []
    for value in values:
        if type(value) is not str:
            raise ValueError("phase coordinate must be canonical hexadecimal text")
        number = float.fromhex(value)
        if not math.isfinite(number) or number.hex() != value:
            raise ValueError("phase coordinate must represent a finite binary64 value")
        result.append(Q.from_float(number))
    if result[0] != 0:
        raise ValueError("phase gauge must be zero")
    return tuple(result)


@pytest.fixture(scope="module")
def evidence():
    with forbid_sine_regeneration():
        geometry = _derive_organized_external_interaction()
        edges = tuple((a, b) for a, b, _ in geometry.weighted_edges)
        branches = tuple(
            [-int(a % 9 == 8) for a, _ in edges[:54]]
            + [-int(a // 9 == 4) for a, _ in edges[54:99]]
        )

        def load(name):
            return json_loads((EVIDENCE / name).read_bytes())

        return (
            load("attempt.json"),
            load("outcome.json"),
            geometry,
            edges,
            branches,
            tuple(
                (load(f"start-{i:02d}.json"), load(f"candidate-{i:02d}.json"))
                for i in range(24)
            ),
        )


def test_source_association_uses_committed_bytes_without_execution(evidence):
    metadata = evidence[0]
    assert metadata["source_revision"] == REVISION
    if shutil.which("git") is None:
        pytest.skip("Git is required only for frozen committed-byte association")
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{REVISION}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if available.returncode:
        pytest.skip("Frozen revision is absent from this shallow or archive checkout")
    for path, expected in metadata["source_blobs"].items():
        data = subprocess.check_output(["git", "show", f"{REVISION}:{path}"], cwd=ROOT)
        payload = f"blob {len(data)}\0".encode() + data
        assert hashlib.sha1(payload).hexdigest() == expected
    tree = subprocess.check_output(
        ["git", "rev-parse", f"{REVISION}:src"], cwd=ROOT, text=True
    ).strip()
    assert metadata["source_tree"] == tree


def test_frozen_inventory_support_and_runtime_are_retained(evidence):
    metadata, outcome, _, edges, branches, records = evidence
    assert outcome == {
        "verdict": "finite_search_without_certificate",
        "completed": list(range(24)),
    }
    assert len(tuple(EVIDENCE.glob("start-*.json"))) == 24
    assert len(tuple(EVIDENCE.glob("candidate-*.json"))) == 24
    assert type(metadata["edges"]) is list
    assert all(type(edge) is list and len(edge) == 2 for edge in metadata["edges"])
    assert all(type(node) is int for edge in metadata["edges"] for node in edge)
    assert type(metadata["marked_branches"]) is list
    assert all(type(branch) is int for branch in metadata["marked_branches"])
    assert type(metadata["node_order"]) is list
    assert all(type(node) is int for node in metadata["node_order"])
    assert tuple(map(tuple, metadata["edges"])) == edges
    assert tuple(metadata["marked_branches"]) == branches
    assert metadata["node_order"] == list(range(54))
    assert metadata["maximum_attempts"] == len(records) == 24
    assert metadata["seed"] == 20261010
    for key in ("python", "numpy", "scipy", "platform", "numeric_backend"):
        assert type(metadata[key]) is str and metadata[key]
    assert all(value == "1" for value in metadata["thread_environment"].values())
    assert not (EVIDENCE / "incomplete.json").exists()


def test_every_start_endpoint_and_budget_is_preserved(evidence):
    for index, (start, record) in enumerate(evidence[-1]):
        assert type(record["number"]) is int and record["number"] == index
        assert start["number"] == index
        assert record["donor_origin_index"] == start["donor_origin_index"] == index // 6
        assert (
            record["amplitude"]
            == start["amplitude"]
            == (0, 0.125, 0.25, 0.5, 1, 1.5)[index % 6]
        )
        assert 0 < record["perturbation_scale"] == start["perturbation_scale"] <= 1
        assert _phases(start["phase_hex"]) == _phases(record["initial"]["phase_hex"])
        _phases(record["endpoint"]["phase_hex"])
        assert type(record["value_gradient_calls"]) is int
        assert 0 < record["value_gradient_calls"] <= 5000
        assert type(record["optimizer"]["iterations"]) is int
        assert 0 <= record["optimizer"]["iterations"] <= 500
        assert record["polish"] is None and record["certificate"] is None


@pytest.mark.parametrize("index", range(24))
def test_endpoint_fields_windings_and_nonstationarity_are_rebuilt(evidence, index):
    _, _, geometry, edges, branches, records = evidence
    endpoint = records[index][1]["endpoint"]
    exact = _phases(endpoint["phase_hex"])
    phase = tuple(map(float, exact))
    current, hessian, costs = [0.0] * 54, np.zeros((54, 54)), []
    for a, b in edges:
        gap = phase[b] - phase[a]
        flow, weight = math.sin(gap), math.cos(gap)
        current[a] += flow
        current[b] -= flow
        hessian[a, a] += weight
        hessian[b, b] += weight
        hessian[a, b] -= weight
        hessian[b, a] -= weight
        costs.append(1 - weight)
    residual = max(abs(value) for value in current[1:])
    slack = (
        math.pi
        - 1 / 1024
        - max(
            abs(phase[b] - phase[a] - math.tau * branch)
            for (a, b), branch in zip(edges[:99], branches)
        )
    )
    assert endpoint["potential"] == pytest.approx(math.fsum(costs), abs=1e-11)
    assert endpoint["gradient_infinity_norm"] == pytest.approx(residual, abs=1e-11)
    assert endpoint["tightened_chart_slack"] == pytest.approx(slack, abs=1e-12)
    assert endpoint["grounded_hessian_minimum_eigenvalue"] == pytest.approx(
        np.linalg.eigvalsh(hessian[1:, 1:])[0], abs=1e-10
    )
    assert abs(slack) < 1e-9
    pi = pi_interval()
    for (a, b), branch in zip(edges[:99], branches):
        gap = I(exact[b] - exact[a]) - 2 * branch * pi
        assert -pi.lo < gap.lo <= gap.hi < pi.lo
    for cycle in geometry.internal_cycles + geometry.outer_cycles:
        winding = (
            math.fsum(
                math.remainder(phase[b] - phase[a], math.tau)
                for a, b in zip(cycle, cycle[1:] + cycle[:1])
            )
            / math.tau
        )
        assert winding == pytest.approx(1, abs=1e-13)
    node = max(range(1, 54), key=lambda i: abs(current[i]))
    witness = I(0)
    for a, b in edges:
        if a == node:
            witness += sin(I(exact[b] - exact[a]))
        elif b == node:
            witness -= sin(I(exact[b] - exact[a]))
    assert not witness.contains(0)
    assert min(abs(witness.lo), abs(witness.hi)) > Q(7, 10000)


@pytest.mark.parametrize("invalid", [True, None, 0.0, "nan", "inf", "0x1p-99999"])
def test_hex_reader_rejects_invalid_or_lost_physical_values(invalid):
    values = [float(0).hex()] * 54
    values[1] = invalid
    with pytest.raises(ValueError):
        _phases(values)
