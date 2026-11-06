"""Portable saved-receipt arithmetic; no historical producers or kernels run."""

import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction as F

import pytest

from benchmarks import thol_child_distortion_audit as audit
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.forcing_realization import NonEpiForcingObservation
from tnfr.physics.support_transport import _from_data


def _mv(matrix, vector):
    return tuple(
        sum((a * b for a, b in zip(row, vector, strict=True)), F(0)) for row in matrix
    )


def _mm(left, right):
    return tuple(
        tuple(
            sum((a * b for a, b in zip(row, column, strict=True)), F(0))
            for column in zip(*right, strict=True)
        )
        for row in left
    )


def _identity(n):
    return tuple(tuple(F(i == j) for j in range(n)) for i in range(n))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _fixture(n=4, defects=False):
    """Declare detached affine receipts, without invoking a reset producer.

    Values are synthetic rational test data, not evidence of native execution.
    Child capacities equal one and children have only parent neighbors.
    """
    half = n // 2
    undirected = {(i, (i + 1) % half) for i in range(half)}
    undirected |= {(i, i + half) for i in range(half)}
    undirected |= {((i + 1) % half, i + half) for i in range(0, half, 2)}
    undirected = {tuple(sorted(pair)) for pair in undirected if pair[0] != pair[1]}
    edges = tuple(
        sorted((i, j, F(1)) for pair in undirected for i, j in (pair, pair[::-1]))
    )
    support = tuple(tuple(j for i, j, _ in edges if i == k) for k in range(n))
    nu, zero = (F(1),) * n, (F(0),) * n
    a = tuple(
        tuple(
            F(i == j) - (F(1, len(support[i])) if j in support[i] else F(0))
            for j in range(n)
        )
        for i in range(n)
    )
    base = tuple(F((i % 4) + 1, 8) for i in range(n))
    delta = tuple(F(1, 16) if i >= half else F(0) for i in range(n))

    def receipt(initial, perturbed):
        geometric = _from_data(tuple(range(n)), edges, support, initial, nu, zero)
        kernel = tuple(
            F(1, 1024) if defects and perturbed and i == half else F(0)
            for i in range(n)
        )
        stored = tuple(
            F(-1, 2048) if defects and perturbed and i == half + 1 else F(0)
            for i in range(n)
        )
        fresh = _add(geometric.epi_gradient, kernel)
        pressure = _add(fresh, stored)
        snap = _from_data(tuple(range(n)), edges, support, initial, nu, pressure)
        observation = NonEpiForcingObservation(
            snapshot=snap,
            phase=zero,
            epi_weight=F(1),
            forcing=zero,
            phase_gradient=zero,
            normalized_weights=(
                ("phase", F(0)),
                ("epi", F(1)),
                ("vf", F(0)),
                ("topo", F(0)),
            ),
            full_kernel_pressure=fresh,
            kernel_pressure_defect=kernel,
            stored_pressure_residual=stored,
        )
        current, matrix, offset, propagated, rows = initial, _identity(n), zero, [], []
        for i in range(n):
            row = tuple(
                (
                    F(3, 4)
                    if j == i
                    else (F(1, 4 * len(support[i])) if j in support[i] else F(0))
                )
                for j in range(n)
            )
            local = list(_identity(n))
            local[i] = row
            local = tuple(local)
            ideal = _mv(local, current)
            defect = tuple(
                F(1, 4096) if defects and perturbed and i == half and j == i else F(0)
                for j in range(n)
            )
            after = _add(ideal, defect)
            rows.append(
                {
                    "node": i,
                    "control": {
                        "EN_mix": F(1, 4),
                        "neighbor_indices": support[i],
                        "neighbor_mean": sum((current[j] for j in support[i]), F(0))
                        / len(support[i]),
                    },
                    "configuration": {"GLYPH_FACTORS": {"EN_mix": 0.25}},
                    "row": row,
                    "offset": F(0),
                    "before_epi": current,
                    "after_epi": after,
                    "kernel_evaluation_defect": defect[i],
                    "clipping_defect": F(0),
                    "local_defect": defect,
                }
            )
            propagated = [_mv(local, value) for value in propagated] + [defect]
            matrix = _mm(local, matrix)
            current = after
        total = tuple(sum((row[i] for row in propagated), F(0)) for i in range(n))
        integration = tuple(
            F(-1, 8192) if defects and perturbed and i == half + 1 else F(0)
            for i in range(n)
        )
        post = tuple(
            F(1, 16384) if defects and perturbed and i == half else F(0)
            for i in range(n)
        )
        xi = _add(_add(current, tuple(F(1, 4) * p for p in pressure)), integration)
        xf = _add(xi, post)
        ideal_endpoint = _sub(
            _mv(matrix, initial), tuple(F(1, 4) * v for v in _mv(a, initial))
        )
        report = {
            "glyph": "EN",
            "nodes": tuple(range(n)),
            "start_time": F(9, 4),
            "end_time": F(5, 2),
            "vectors": {"x0": initial, "xg": current, "xi": xi, "xf": xf},
            "reset": {
                "S": matrix,
                "c": offset,
                "local_rows": rows,
                "propagated_local_defects": propagated,
                "total_reset_defect": total,
                "identity_residual": zero,
            },
            "generation": {"A": a, "b": zero, "observation": asdict(observation)},
            "runtime": {
                "ideal_endpoint": ideal_endpoint,
                "pressure_term": tuple(F(1, 4) * v for v in _add(kernel, stored)),
                "integration_remainder": integration,
                "postintegration_change": post,
                "identity_residual": zero,
                "pre_generated_pressure_retained": True,
            },
        }
        return audit._payload(report), snap

    left, snap = receipt(base, False)
    right, _ = receipt(_add(base, delta), True)
    original = derive_forced_support_balance(snap, epi_weight=1, forcing=zero)
    return left, right, original, tuple(range(half, n))


def _energy(values, metric):
    mean = sum((h * x for h, x in zip(metric, values, strict=True)), F(0)) / sum(metric)
    centered = tuple(x - mean for x in values)
    return mean, sum(
        (h * x * x / 2 for h, x in zip(metric, centered, strict=True)), F(0)
    )


@pytest.mark.parametrize("defects", (False, True))
def test_exact_per_write_and_integration_telescope(defects):
    left, right, original, children = _fixture(defects=defects)
    result = audit.audit_pair(left, right, original, children)
    metric = tuple(original.metric_weights[i] for i in children)
    total = F(0)
    for row in result["local_EN_writes"]:
        before, after = row["delta_before"], row["delta_after"]
        assert row["paired_increment"] == _sub(
            row["perturbed_increment"], row["control_increment"]
        )
        assert all(
            value == 0
            for i, value in enumerate(row["paired_increment"])
            if i != row["node"]
        )
        u, v = (tuple(value[i] for i in children) for value in (before, after))
        change = _energy(v, metric)[1] - _energy(u, metric)[1]
        assert change == row["budget"]["variance_change"]
        assert (
            change
            == row["budget"]["variance_defect_linear_term"]
            + row["budget"]["variance_defect_quadratic_term"]
        )
        total += change
    assert total == result["reset_budget"]["variance_change"]
    total += (
        result["integration_budget"]["variance_change"]
        + result["postintegration_budget"]["variance_change"]
    )
    assert (
        total
        == result["states"]["xf"]["centered_H_energy"]
        - result["states"]["x0"]["centered_H_energy"]
    )
    assert (
        result["telescope"]["variance_residual"]
        == result["telescope"]["mass_residual"]
        == 0
    )
    for key in ("reset_budget", "integration_budget", "postintegration_budget"):
        assert (
            result[key]["mass_identity_residual"]
            == result[key]["variance_identity_residual"]
            == 0
        )


def test_held_generation_pressure_is_not_refreshed_at_reset_endpoint():
    left, right, original, children = _fixture(defects=True)
    result = audit.audit_pair(left, right, original, children)
    g = result["integration_budget"]
    lag = result["pressure_work"]["held_generation_lag"]["pressure"]
    assert any(lag)
    a = tuple(tuple(F(x) for x in row) for row in left["generation"]["A"])
    assert lag == _mv(a, _sub(result["delta"]["xg"], result["delta"]["x0"]))
    expected = tuple(
        sum((part["pressure"][i] for part in result["pressure_work"].values()), F(0))
        for i in range(4)
    )
    assert expected == g["balance"]["stored_pressure_defect"]
    assert (
        sum((part["variance_work"] for part in result["pressure_work"].values()), F(0))
        == F(1, 4) * g["balance"]["variance_defect_rate"]
    )
    # Actual xi uses pressure generated at delta0; a refreshed endpoint differs.
    refreshed = _sub(
        result["delta"]["xg"], tuple(F(1, 4) * x for x in _mv(a, result["delta"]["xg"]))
    )
    assert refreshed != result["delta"]["xi"]


def test_complete_four_mode_gram_reconstruction_and_mean_transfer():
    left, right, original, children = _fixture(defects=True)
    result = audit.audit_pair(left, right, original, children)
    modes = result["mode_diagnostic"]
    vectors = modes["centered_child_contributions"]
    metric = tuple(original.metric_weights[i] for i in children)
    assert len(vectors) == 8 and len(modes["gram_terms"]) == 36
    for row in modes["gram_terms"]:
        inner = sum(
            (
                h * a * b
                for h, a, b in zip(
                    metric, vectors[row["left"]], vectors[row["right"]], strict=True
                )
            ),
            F(0),
        )
        assert inner == row["inner_product"]
        assert row["energy_contribution"] == (
            inner / 2 if row["left"] == row["right"] else inner
        )
    assert (
        sum((row["energy_contribution"] for row in modes["gram_terms"]), F(0))
        == result["states"]["xf"]["centered_H_energy"]
    )
    assert modes["child_mean_contrast_transport_to_shape_zero"]
    assert any(modes["centered_S_child_indicator"])
    assert modes["centered_minus_hA_child_indicator"] == (F(0),) * len(children)
    assert modes["energy_residual"] == 0


@pytest.mark.parametrize(
    "mutation",
    (
        "row_order",
        "row_value",
        "staged_epi",
        "local_defect",
        "propagated",
        "matrix",
        "offset",
        "generation_epi",
        "kernel_defect",
        "pressure_retained",
        "integration_remainder",
        "time",
        "configuration",
        "neighbor_order",
    ),
)
def test_inconsistent_retained_receipt_rejected(mutation):
    left, right, original, children = _fixture()
    row = right["reset"]["local_rows"][0]
    if mutation == "row_order":
        right["reset"]["local_rows"].reverse()
    elif mutation == "row_value":
        row["row"][0] = "1/3"
    elif mutation == "staged_epi":
        row["after_epi"][1] = "999"
    elif mutation == "local_defect":
        row["local_defect"][0] = "1"
    elif mutation == "propagated":
        right["reset"]["propagated_local_defects"][0][0] = "1"
    elif mutation == "matrix":
        right["reset"]["S"][0][0] = "1"
    elif mutation == "offset":
        row["offset"] = "1"
    elif mutation == "generation_epi":
        right["generation"]["observation"]["snapshot"]["epi"][0] = "99"
    elif mutation == "kernel_defect":
        right["generation"]["observation"]["kernel_pressure_defect"][0] = "1"
    elif mutation == "pressure_retained":
        right["runtime"]["pre_generated_pressure_retained"] = False
    elif mutation == "integration_remainder":
        right["runtime"]["integration_remainder"][0] = "1"
    elif mutation == "time":
        right["end_time"] = "3"
    elif mutation == "configuration":
        row["configuration"]["GLYPH_FACTORS"]["EN_mix"] = 0.5
    elif mutation == "neighbor_order":
        row["control"]["neighbor_indices"].reverse()
    with pytest.raises((ValueError, RuntimeError)):
        audit.audit_pair(left, right, original, children)


def _saved_reports():
    left, right, original, children = _fixture(16)
    binding = {
        key: {"sha256": "a" * 64, "historical_manifest": {"claim_id": key}}
        for key in ("window", "native")
    }
    first = {
        "historical_inputs": deepcopy(binding),
        "original_reference": audit._payload(asdict(original)),
        "branches": [
            {"branch": name, "resets": [row]}
            for name, row in zip(audit.BRANCHES, (left, right))
        ],
    }
    metric = tuple(original.metric_weights[i] for i in children)
    endpoints = []
    for time, key in ((F(9, 4), "x0"), (F(5, 2), "xf")):
        x, y = (
            tuple(F(row["vectors"][key][i]) for i in children) for row in (left, right)
        )
        mean, energy = _energy(_sub(y, x), metric)
        delta = _sub(y, x)
        paired = {
            "epi_difference": delta,
            "weighted_mean_offset": mean,
            "centered_epi_difference": tuple(v - mean for v in delta),
            "centered_H_energy": energy,
            "full_H_energy": sum(
                (h * v * v / 2 for h, v in zip(metric, delta, strict=True)), F(0)
            ),
            "full_epi_equal": not any(delta),
            "shape_equal_modulo_uniform_offset": not any(v - mean for v in delta),
        }
        endpoints.append(
            {"time": time, "control_epi": x, "perturbed_epi": y, "paired": paired}
        )
    second = {
        "historical_inputs": binding,
        "nodes": original.source.nodes,
        "full_metric_weights": original.metric_weights,
        "actual_lineage": {
            "pairs": tuple((i, i + 8) for i in range(8)),
            "children": children,
        },
        "regions": [
            {
                "label": "actual_children",
                "nodes": children,
                "full_node_indices": children,
                "metric_weights": metric,
                "endpoints": endpoints,
            }
        ],
    }
    return first, audit._payload(second)


def test_saved_reports_bind_lineage_metric_and_both_endpoints_without_kernels(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("no old producer or kernel may run")

    monkeypatch.setattr(audit.reset, "run_study", forbidden)
    monkeypatch.setattr(audit.reset, "audit_reset_step", forbidden)
    monkeypatch.setattr(audit.reset, "neighbor_epi_blend_value", forbidden)
    first, second = _saved_reports()
    result = audit.audit_saved(first, second)
    assert (
        result["native_calls"]
        == result["kernel_calls"]
        == result["forcing_capture_calls"]
        == 0
    )
    assert len(result["audit"]["local_EN_writes"]) == 16
    assert len(result["audit"]["children"]) == 8


@pytest.mark.parametrize(
    "mutation",
    ("hash", "manifest", "lineage", "metric", "endpoint", "paired", "branches"),
)
def test_cross_report_bindings_reject_substitution(mutation):
    first, second = _saved_reports()
    if mutation == "hash":
        second["historical_inputs"]["native"]["sha256"] = "b" * 64
    elif mutation == "manifest":
        second["historical_inputs"]["window"]["historical_manifest"][
            "claim_id"
        ] = "other"
    elif mutation == "lineage":
        second["actual_lineage"]["pairs"][0][1] = 9
    elif mutation == "metric":
        second["full_metric_weights"][8] = "999"
    elif mutation == "endpoint":
        second["regions"][0]["endpoints"][0]["control_epi"][0] = "999"
    elif mutation == "paired":
        second["regions"][0]["endpoints"][1]["paired"]["centered_H_energy"] = "0"
    elif mutation == "branches":
        first["branches"].reverse()
    with pytest.raises(ValueError):
        audit.audit_saved(first, second)


def _manifest(claim):
    manifest = audit.CoreExperimentManifest(
        claim_id=claim,
        git_sha="a" * 40,
        source_dirty=False,
        versions={"python": "synthetic"},
        graph_construction="Synthetic detached fixture",
        capacity_specification="Declared positive",
        solver="Fixture arithmetic",
        timestep=None,
        seed=None,
        result_status=audit.ClaimStatus.DERIVED,
        operator_sequence=(),
        telemetry=("fixture",),
        controls=("not native execution",),
        artifacts=("fixture.json",),
    )
    manifest.validate_for_admission()
    return manifest.to_dict()


def test_loader_authenticates_bytes_and_cli_protects_both_inputs(tmp_path, monkeypatch):
    first, second = _saved_reports()
    first["manifest"] = _manifest("O3.b-retained-EN-AL-reset-accounting")
    second["manifest"] = _manifest("O3.a-retained-regional-paired-response")
    paths, hashes = [], []
    for i, row in enumerate((first, second)):
        path = tmp_path / f"input{i}.json"
        path.write_text(json.dumps(audit._payload(row)), encoding="utf8")
        paths.append(path)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    result = audit.run_study(
        *paths, expected_reset_sha256=hashes[0], expected_recovery_sha256=hashes[1]
    )
    assert result["historical_inputs"]["recovery"]["sha256"] == hashes[1]
    with pytest.raises(ValueError, match="digest"):
        audit.run_study(
            *paths, expected_reset_sha256="0" * 64, expected_recovery_sha256=hashes[1]
        )
    for path in paths:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "audit",
                "--reset-input",
                str(paths[0]),
                "--recovery-input",
                str(paths[1]),
                "--output",
                str(path),
            ],
        )
        with pytest.raises(ValueError, match="overwrite"):
            audit.main()
