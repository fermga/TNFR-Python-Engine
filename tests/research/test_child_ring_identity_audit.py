"""Portable recorded-state controls; no historical artifact or operator execution."""

import math
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction as F

import pytest

from benchmarks import thol_child_ring_identity_audit as audit
from benchmarks.thol_pressure_feedback import _payload
from tnfr.physics.forced_support import (
    derive_forced_support_balance,
    observe_forced_support_step,
)
from tnfr.physics.forcing_realization import (
    NonEpiForcingObservation,
    decompose_non_epi_forcing,
)
from tnfr.physics.support_transport import _from_data

NODES = ("a", "b", "c", "outside")
REGION = NODES[:3]
CAPACITY = (F(1), F(2), F(1), F(1))
PHASE = (F(0), F(1, 4), F(1, 2), F(3, 4))


def _snapshot(epi=(0, 1, 2, 5), *, ring=True, pressure=None):
    """Declare primitive data; the supplied phase channel is disabled."""
    edges = [(0, 1, F(1)), (1, 2, F(1)), (2, 3, F(2))]
    if ring:
        edges.append((0, 2, F(1)))
    directed, support = [], [set() for _ in NODES]
    for i, j, weight in edges:
        directed.extend(((i, j, weight), (j, i, weight)))
        support[i].add(j)
        support[j].add(i)
    rows = tuple(tuple(sorted(row)) for row in support)
    source = _from_data(NODES, directed, rows, epi, CAPACITY, (0,) * 4)
    if pressure is None:
        pressure = tuple(F(float(value)) for value in source.epi_gradient)
    return _from_data(NODES, directed, rows, epi, CAPACITY, pressure)


def _state(source, time=1.0):
    return {
        "time": time,
        "nodes": list(source.nodes),
        "epi": list(map(float, source.epi)),
        "capacity": list(map(float, source.capacity)),
        "pressure": list(map(float, source.stored_pressure)),
        "phase": list(map(float, PHASE)),
        "edges": [
            [NODES[i], NODES[j], {"weight": float(weight), "length": 2.0}]
            for i, j, weight in source.conductance
            if i < j
        ],
    }


def _capture(source):
    zeros = (F(0),) * len(NODES)
    observation = NonEpiForcingObservation(
        snapshot=source,
        phase=PHASE,
        epi_weight=F(1),
        forcing=zeros,
        phase_gradient=zeros,
        normalized_weights=(
            ("phase", F(0)),
            ("epi", F(1)),
            ("vf", F(0)),
            ("topo", F(0)),
        ),
        full_kernel_pressure=source.stored_pressure,
        kernel_pressure_defect=tuple(
            p - g for p, g in zip(source.stored_pressure, source.epi_gradient)
        ),
        stored_pressure_residual=zeros,
    )
    return _payload(
        {
            "observation": asdict(observation),
            "components": decompose_non_epi_forcing(observation),
        }
    )


def _observe(state, capture):
    return audit.observe_recorded_cohort(
        state,
        capture,
        REGION,
        phase_gate=math.pi / 2,
    )


def test_cohort_keeps_boundary_full_metric_triad_and_offline_tetrad(monkeypatch):
    source = _snapshot()
    state, capture = _state(source), _capture(source)
    preserved = deepcopy((state, capture))
    seen = []
    owner = audit.compute_structural_potential

    def record(graph, **kwargs):
        seen.append((tuple(graph), graph["c"]["outside"]["length"], kwargs))
        return owner(graph, **kwargs)

    monkeypatch.setattr(audit, "compute_structural_potential", record)
    result = _observe(state, capture)
    assert (state, capture) == preserved
    assert seen == [(NODES, 2.0, {"alpha": 2.0})]
    assert result["triad"] == {
        "epi": source.epi[:3],
        "capacity": CAPACITY[:3],
        "phase": PHASE[:3],
    }
    balance = result["regional"]["balance"]
    # The outside edge contributes to H_c=4, the cut flux and the regional mean.
    assert balance["strengths"] == (2, 2, 4, 2)
    assert balance["metric_weights"] == (2, 1, 4, 2)
    assert balance["mean"] == F(9, 7)
    assert balance["internal_dissipation"] == 6
    assert balance["mass_boundary_rate"] == 6
    assert balance["variance_boundary_rate"] == F(30, 7)
    assert balance["stored_variance_rate"] == F(-12, 7)
    assert balance["variance_identity_residual"] == 0
    tetrad = result["offline_tetrad"]
    assert tetrad["nodes"] == NODES
    assert all(len(tetrad[key]) == 4 for key in ("phi_s", "grad_phi", "curv_phi"))
    assert tetrad["xi_c"]["method"]
    assert "distance_weighting" in tetrad["xi_c"]
    assert "fit_available" in tetrad["xi_c"]
    assert "not historical telemetry" in tetrad["scope"]
    assert "dynamical feedback" in tetrad["scope"]


@pytest.mark.parametrize(
    "changed", ("epi", "capacity", "phase", "pressure", "edge", "cache")
)
def test_cohort_rejects_raw_capture_or_cached_snapshot_disagreement(changed):
    source = _snapshot()
    state, capture = _state(source), _capture(source)
    if changed == "edge":
        state["edges"][0][2]["weight"] = 2.0
    elif changed == "cache":
        capture["observation"]["snapshot"]["dirichlet_energy"] = "99"
    else:
        state[changed][0] += 0.125
    with pytest.raises(ValueError):
        _observe(state, capture)


def test_absent_cycle_is_undefined_and_distinct_from_defined_zero_winding():
    ring, path = _snapshot(), _snapshot(ring=False)
    defined = _observe(_state(ring), _capture(ring))["winding"]
    absent = _observe(_state(path), _capture(path))["winding"]
    assert defined["status"] == "defined" and defined["winding"] == 0
    assert defined["cycle_exists"] is True
    assert absent["status"] == "undefined" and absent["winding"] is None
    assert absent["cycle_exists"] is False
    assert absent["raw_winding"] is None


def _finite_records():
    """Two declared Euler identities, not receipts of a live solver execution."""
    source = _snapshot()
    reference = derive_forced_support_balance(source, epi_weight=1, forcing=(0,) * 4)
    segments, steps = [], []
    before = source
    for index in range(2):
        duration = F(1, 4)
        epi = tuple(x + duration * r for x, r in zip(before.epi, before.rate))
        after = _snapshot(epi, pressure=before.stored_pressure)
        interval = {
            "exact_start_time": str(F(1) + index * duration),
            "exact_end_time": str(F(1) + (index + 1) * duration),
            "exact_duration": str(duration),
            "duration": float(duration),
        }
        segments.append(
            {
                "index": index,
                "method": "euler",
                "interval": interval,
                "before_support": asdict(before),
                "after_support": asdict(after),
            }
        )
        steps.append(
            asdict(observe_forced_support_step(reference, before, after, duration))
        )
        before = _snapshot(after.epi)
    branch = _payload(
        {
            "continuation_flow": {
                "before": _state(source),
                "after": _state(before, 1.5),
                "segments": segments,
            },
            "continuation_steps": steps,
        }
    )
    return branch, reference


def test_complete_finite_segments_telescope_with_pressure_refresh_and_no_mutation():
    branch, reference = _finite_records()
    preserved = deepcopy(branch)
    segments = branch["continuation_flow"]["segments"]
    assert (
        segments[0]["after_support"]["stored_pressure"]
        != segments[1]["before_support"]["stored_pressure"]
    )
    result = audit._finite_budgets(branch, reference, REGION)
    assert branch == preserved
    rows = [item["regional_budget"] for item in result["segments"]]
    assert rows[0]["balance"]["variance"] == F(19, 7)
    assert rows[0]["balance"]["variance_boundary_rate"] == F(30, 7)
    assert rows[0]["variance_drift_term"] == F(-3, 7)
    assert all(row["state_defect"] == (0,) * 4 for row in rows)
    assert all(
        row["mass_identity_residual"] == row["variance_identity_residual"] == 0
        for row in rows
    )
    assert result["variance_change"] == sum(row["variance_change"] for row in rows)
    assert result["variance_telescope_residual"] == 0
    assert "not exact continuous-time integrated fluxes" in result["scope"]


@pytest.mark.parametrize(
    "changed",
    (
        "missing_step",
        "index",
        "method",
        "interval_gap",
        "duration",
        "source",
        "endpoint",
        "endpoint_time",
        "continuity",
        "old_budget",
    ),
)
def test_finite_budgets_reject_incomplete_or_incompatible_retained_evidence(changed):
    branch, reference = _finite_records()
    flow = branch["continuation_flow"]
    segments = flow["segments"]
    if changed == "missing_step":
        branch["continuation_steps"].pop()
    elif changed == "index":
        segments[1]["index"] = 0
    elif changed == "method":
        segments[0]["method"] = "rk4"
    elif changed == "interval_gap":
        segments[1]["interval"]["exact_start_time"] = "3/2"
    elif changed == "duration":
        segments[0]["interval"]["duration"] = 0.5
    elif changed == "source":
        flow["before"]["epi"][0] += 0.125
    elif changed == "endpoint":
        flow["after"]["epi"][0] += 0.125
    elif changed == "endpoint_time":
        flow["after"]["time"] = 1.75
    elif changed == "continuity":
        # Internally consistent replacement snapshot: the inter-segment check must fail.
        saved = segments[1]["before_support"]
        shifted = tuple(F(value) + F(1, 8) for value in saved["epi"])
        segments[1]["before_support"] = _payload(asdict(_snapshot(shifted)))
    else:
        branch["continuation_steps"][0]["relative_energy_budget"][
            "energy_change"
        ] = "99"
    with pytest.raises(ValueError):
        audit._finite_budgets(branch, reference, REGION)


@pytest.mark.parametrize("scale", (F(3), F(-2)))
def test_shape_projection_separates_uniform_offset_signed_scale_and_amplitude(scale):
    first, metric = (F(0), F(1), F(3)), (F(1), F(2), F(1))
    current = tuple(7 + scale * value for value in first)
    observed = audit.observe_shape_retention(first, current, metric)
    assert observed["signed_amplitude_projection"] == scale
    assert observed["initial_squared_norm"] == F(19, 4)
    assert observed["current_squared_norm"] == scale**2 * F(19, 4)
    assert observed["orthogonal_residual_squared_norm"] == 0
    assert observed["relative_squared_residual"] == 0
    assert observed["exact_collinearity"] is True
    assert "acceptance threshold" in observed["scope"]


def test_shape_projection_retains_exact_transverse_difference():
    # These two centered P3 directions are H-orthogonal, with H-norms 2 and 4.
    observed = audit.observe_shape_retention((-1, 0, 1), (-1, -1, 3), (1, 2, 1))
    assert observed["signed_amplitude_projection"] == 2
    assert observed["current_squared_norm"] == 12
    assert observed["orthogonal_residual_squared_norm"] == 4
    assert observed["relative_squared_residual"] == F(1, 3)
    assert observed["exact_collinearity"] is False


def test_uniform_initial_or_current_form_leaves_unavailable_normalization_explicit():
    initial_uniform = audit.observe_shape_retention((2, 2, 2), (0, 1, 2), (1, 2, 1))
    assert initial_uniform["initial_squared_norm"] == 0
    assert initial_uniform["signed_amplitude_projection"] is None
    assert initial_uniform["orthogonal_residual_squared_norm"] is None
    assert initial_uniform["exact_collinearity"] is None
    current_uniform = audit.observe_shape_retention((0, 1, 2), (2, 2, 2), (1, 2, 1))
    assert current_uniform["current_squared_norm"] == 0
    assert current_uniform["signed_amplitude_projection"] == 0
    assert current_uniform["orthogonal_residual_squared_norm"] == 0
    assert current_uniform["relative_squared_residual"] is None


@pytest.mark.parametrize(
    "initial,current,metric",
    (
        ((), (), ()),
        ((0, 1), (1,), (1, 1)),
        ((0, 1), (1, 2), (1,)),
        ((0, 1), (1, 2), (1, 0)),
        ((0, 1), (1, 2), (1, -1)),
    ),
)
def test_shape_projection_rejects_mismatched_or_nonpositive_metric(
    initial, current, metric
):
    with pytest.raises(ValueError):
        audit.observe_shape_retention(initial, current, metric)
