"""Observe THOL acceleration feedback, pressure refresh and isolated births.

C8 acceleration comes from three executor-recorded physical samples after
actual IL/OZ applications. Public, staged and ordinary selector routes retain
their own semantics. A separate P2 birth case has explicitly prepared history.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.capacity_localization import build_cycle  # noqa: E402
from tnfr.alias import get_attr, set_attr  # noqa: E402
from tnfr.config import inject_defaults  # noqa: E402
from tnfr.constants.aliases import (  # noqa: E402
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.dynamics.integrators import (  # noqa: E402
    DefaultIntegrator,
    update_epi_via_nodal_equation,
)
from tnfr.dynamics.runtime import step  # noqa: E402
from tnfr.operators import (  # noqa: E402
    build_operator_event_schedule,
    build_physical_flow_partition,
    execute_operator_event_schedule,
)
from tnfr.operators.definitions import (  # noqa: E402
    Coherence,
    Dissonance,
    SelfOrganization,
)
from tnfr.operators.factor_contracts import (  # noqa: E402
    resolve_runtime_operator_factors,
)
from tnfr.operators.grammar_dynamics import validate_candidate  # noqa: E402
from tnfr.operators.network_stage import execute_self_organization_stage  # noqa: E402
from tnfr.operators.nodal_equation import compute_d2epi_dt2  # noqa: E402
from tnfr.operators.self_organization import _configured_tau  # noqa: E402
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)
from tnfr.types import Glyph  # noqa: E402

PREPARATION_STEPS = (0.125, 0.375)
NEXT_STEP = 0.25
ROUTES = (
    "baseline",
    "public_held",
    "public_refreshed",
    "staged_refreshed",
    "selector_runtime",
)


def _values(graph, aliases):
    return tuple(
        float(get_attr(data, aliases, 0.0)) for _, data in graph.nodes(data=True)
    )


def _exact(values):
    return tuple(Fraction.from_float(float(value)) for value in values)


def _state(graph):
    return {
        "time": float(graph.graph.get("_t", 0.0)),
        "nodes": tuple(graph),
        "epi": _values(graph, ALIAS_EPI),
        "capacity": _values(graph, ALIAS_VF),
        "phase": _values(graph, ALIAS_THETA),
        "pressure": _values(graph, ALIAS_DNFR),
        "cached_acceleration": _values(graph, ALIAS_D2EPI),
        "edges": tuple((u, v, dict(data)) for u, v, data in graph.edges(data=True)),
        "glyph_history": {
            node: tuple(data.get("glyph_history", ()))
            for node, data in graph.nodes(data=True)
        },
        "physical_epi_history": {
            node: tuple(data.get("epi_time_history", ()))
            for node, data in graph.nodes(data=True)
        },
        "children": {
            node: tuple(data.get("sub_nodes", ()))
            for node, data in graph.nodes(data=True)
        },
        "hierarchy": dict(graph.graph.get("hierarchy", {})),
    }


def _acceleration(graph, node=0):
    samples = tuple(graph.nodes[node]["epi_time_history"])[-3:]
    (t0, x0), (t1, x1), (t2, x2) = tuple(_exact(row) for row in samples)
    h1, h2 = t1 - t0, t2 - t1
    slopes = ((x1 - x0) / h1, (x2 - x1) / h2)
    exact = 2 * (slopes[1] - slopes[0]) / (h1 + h2)
    actual = compute_d2epi_dt2(graph, node, store=False)
    return {
        "physical_samples": samples,
        "exact_secants": slopes,
        "exact_three_point_acceleration": exact,
        "observed_acceleration": actual,
        "cached_acceleration": float(get_attr(graph.nodes[node], ALIAS_D2EPI, 0)),
        "exact_acceleration_arithmetic_residual": (Fraction.from_float(actual) - exact),
    }


def _prepare_recorded_ring():
    graph = build_cycle(8, epi=(1.0,) + (0.5,) * 7)
    inject_defaults(graph)
    Coherence()(graph, 0)
    Dissonance()(graph, 0)
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(sum(PREPARATION_STEPS),),
    )
    partition = build_physical_flow_partition(
        schedule.intervals[0],
        PREPARATION_STEPS,
    )
    execution = execute_operator_event_schedule(
        graph,
        schedule,
        method="euler",
        physical_flow_partitions=(partition,),
    )
    evidence = execution.physical_flow_partition_evidence[0]
    preparation = {
        "actual_prefix": tuple(graph.nodes[0]["glyph_history"]),
        "physical_steps": PREPARATION_STEPS,
        "acceleration": _acceleration(graph),
        "boundaries": [
            {
                "time": item.time,
                "epi": item.after.epi,
                "pressure": item.after.delta_nfr,
                "pressure_only_refresh": item.nonpressure_state_preserved,
            }
            for item in evidence.boundary_observations
        ],
        "segment_methods": tuple(
            item.resolved_method for item in evidence.segment_flow_evidence
        ),
        "clipping_applied": tuple(
            item.clipping_applied for item in evidence.segment_flow_evidence
        ),
        "scope": "Three physical samples from one existing event-free partition",
    }
    return graph, preparation


def _integration_record(before, after, graph):
    dt = Fraction.from_float(NEXT_STEP)
    x, nu, pressure, output = (
        _exact(values)
        for values in (
            before["epi"],
            before["capacity"],
            before["pressure"],
            after["epi"],
        )
    )
    predicted = tuple(a + dt * b * c for a, b, c in zip(x, nu, pressure, strict=True))
    low, high = (
        Fraction.from_float(float(graph.graph[name])) for name in ("EPI_MIN", "EPI_MAX")
    )
    return {
        "before": before,
        "after": after,
        "duration": NEXT_STEP,
        "exact_held_input_prediction": predicted,
        "exact_held_input_euler_residual": tuple(
            actual - ideal for actual, ideal in zip(output, predicted, strict=True)
        ),
        "held_proposals_strictly_inside_hard_bounds": all(
            low < value < high for value in predicted
        ),
        "clip_mode": graph.graph["CLIP_MODE"],
        "scope": "Shared classical nodal Euler with explicit held pressure and Gamma=0",
    }


class _RecordingIntegrator(DefaultIntegrator):
    """Read-only instrumentation around the ordinary default integrator."""

    def __init__(self):
        self.records = []

    def integrate(self, graph, *, dt, t, method, n_jobs):
        before = _state(graph)
        super().integrate(graph, dt=dt, t=t, method=method, n_jobs=n_jobs)
        self.records.append(_integration_record(before, _state(graph), graph))


def _target_zero_thol(_graph, node):
    """Declared fixed selector policy; no claim of autonomous THOL discovery."""
    return "THOL" if node == 0 else None


def run_recorded_thol_case(route, *, poison_cache=False):
    """Compare one admitted THOL operation and its immediate pressure consumer."""
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}")
    graph, preparation = _prepare_recorded_ring()
    if poison_cache:
        set_attr(graph.nodes[0], ALIAS_D2EPI, 9.0)
    before = _state(graph)
    acceleration = _acceleration(graph)
    admission = validate_candidate(graph, 0, "THOL")
    factor = resolve_runtime_operator_factors(
        graph.graph.get("GLYPH_FACTORS"),
        Glyph.THOL,
        graph.graph,
    )["THOL_accel"]
    birth_threshold = _configured_tau(graph.graph, {})
    refreshed = None
    stage = None
    if route == "selector_runtime":
        integrator = _RecordingIntegrator()
        graph.graph.update(
            integrator=integrator,
            glyph_selector=_target_zero_thol,
            INTEGRATOR_METHOD="euler",
        )
        step(graph, dt=NEXT_STEP, use_Si=False, apply_glyphs=True)
        integration = integrator.records[0]
        raw = integration["before"]
    else:
        if route == "staged_refreshed":
            observed = []

            def capture_and_refresh(current):
                observed.append(_state(current))
                default_compute_delta_nfr(current)
                observed.append(_state(current))

            result = execute_self_organization_stage(
                graph,
                SelfOrganization(),
                (0,),
                compute_delta_nfr=capture_and_refresh,
                collect_metrics=True,
            )
            raw, refreshed = observed
            stage = {
                "schedule": result.schedule,
                "nodes_processed": result.nodes_processed,
            }
        else:
            if route != "baseline":
                SelfOrganization()(graph, 0, collect_metrics=True)
            raw = _state(graph)
            if route == "public_refreshed":
                default_compute_delta_nfr(graph)
                refreshed = _state(graph)
        integration_before = _state(graph)
        update_epi_via_nodal_equation(graph, dt=NEXT_STEP, method="euler")
        integration = _integration_record(integration_before, _state(graph), graph)
    pressure_change = Fraction.from_float(raw["pressure"][0]) - Fraction.from_float(
        before["pressure"][0]
    )
    ideal_change = Fraction.from_float(factor) * Fraction.from_float(
        acceleration["observed_acceleration"]
    )
    return {
        "route": route,
        "preparation": preparation,
        "before": before,
        "acceleration": acceleration,
        "cache_poisoned_for_regression": poison_cache,
        "grammar_admission": {
            "allowed": admission.allowed,
            "candidate": admission.candidate,
            "scope": "Incremental live admission after actual IL and OZ",
        },
        "default_thol_factor": factor,
        "default_birth_threshold": birth_threshold,
        "raw_after_operator": raw,
        "after_refresh": refreshed,
        "stage": stage,
        "integration": integration,
        "whole_route_endpoint": _state(graph),
        "exact_operator_pressure_change": pressure_change,
        "exact_thol_pressure_proposal": ideal_change,
        "exact_pressure_arithmetic_residual": (
            pressure_change - ideal_change if route != "baseline" else Fraction(0)
        ),
        "scope": (
            "Direct/staged calls are single incrementally admitted operations. "
            "The ordinary runtime has a declared selector and a pass-through "
            "integrator observer; later ordinary phase/capacity updates are "
            "retained separately from its integrator endpoint. No changing-node "
            "event-executor or generic feedback convergence is inferred."
        ),
    }


def run_prepared_birth_case():
    """Use the existing P2 birth fixture with prepared physical history."""
    graph = nx.Graph()
    graph.add_node(
        0,
        EPI=0.6,
        nu_f=1.0,
        theta=0.1,
        delta_nfr=0.2,
        epi_time_history=[(0.0, 0.0), (1.0, 0.1), (2.0, 0.6)],
        epi_history=[0.0, 0.1, 0.6],
        glyph_history=["OZ"],
        epi_kind="seed-identity",
    )
    graph.add_node(1, EPI=0.4, nu_f=1.0, theta=0.12, delta_nfr=0.1)
    graph.add_edge(0, 1, weight=1.0, length=1.0)
    graph.graph.update(
        _t=2.0,
        RANDOM_SEED=17,
        GLYPH_HYSTERESIS_WINDOW=64,
        _gamma_spec={"type": "none"},
        GAMMA={"type": "none"},
        use_extended_dynamics=False,
        DT_MIN=0.0,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        compute_delta_nfr=default_compute_delta_nfr,
    )
    default_compute_delta_nfr(graph)
    before = _state(graph)
    acceleration = _acceleration(graph)
    birth_threshold = _configured_tau(graph.graph, {})
    admission = validate_candidate(graph, 0, "THOL")
    SelfOrganization()(graph, 0, collect_metrics=True)
    raw = _state(graph)
    children = tuple(graph.nodes[0].get("sub_nodes", ()))
    default_compute_delta_nfr(graph)
    refreshed = _state(graph)
    update_epi_via_nodal_equation(graph, dt=NEXT_STEP, method="euler")
    return {
        "fixture_source": "tests/operators/test_self_organization_atomicity.py::_graph",
        "preparation_scope": (
            "Physical EPI samples and prior OZ glyph are declared fixture data; "
            "they are not an executed earlier trajectory"
        ),
        "grammar_admission_allowed": admission.allowed,
        "default_birth_threshold": birth_threshold,
        "before": before,
        "acceleration": acceleration,
        "raw_after_operator": raw,
        "after_refresh": refreshed,
        "children": children,
        "child_degrees": tuple(graph.degree(child) for child in children),
        "sub_epi_records": tuple(graph.nodes[0].get("sub_epis", ())),
        "integration": _integration_record(refreshed, _state(graph), graph),
        "scope": (
            "Default metabolic THOL materializes isolated children; hierarchy "
            "membership does not automatically create transport edges"
        ),
    }


def _payload(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_payload(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/research/thol_pressure_feedback.json",
    )
    args = parser.parse_args()
    scope = (
        "src/tnfr",
        "benchmarks/capacity_localization.py",
        "benchmarks/thol_pressure_feedback.py",
    )
    provenance = current_git_source_provenance(ROOT, scope)
    sha, dirty, digest = provenance
    manifest = CoreExperimentManifest(
        claim_id="O3.a-thol-acceleration-pressure-consumption-and-birth",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
        },
        graph_construction=(
            "Unit C8 with EPI bump; separately prepared P2 birth fixture"
        ),
        capacity_specification="Initial unit capacity; default THOL child capacity",
        solver="Existing refreshed event preparation and shared nodal Euler consumers",
        timestep=NEXT_STEP,
        seed=17,
        result_status=ClaimStatus.MEASURED,
        operator_sequence=(
            "actual IL OZ preparation",
            "single THOL via declared routes",
        ),
        telemetry=(
            "physical secants and cached acceleration",
            "pressure before and after refresh",
            "shared-integrator EPI increments",
            "child nodes and transport edges",
        ),
        controls=(
            "held baseline",
            "canonical pressure refresh",
            "prepared birth history",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    report = {
        "manifest": manifest.to_dict(),
        "source_scope": scope,
        "recorded_cases": [run_recorded_thol_case(route) for route in ROUTES],
        "prepared_birth": run_prepared_birth_case(),
        "experimental_status": "No empirical correspondence tested",
    }
    if current_git_source_provenance(ROOT, scope) != provenance:
        raise RuntimeError("source changed while executing the research artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_payload(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote finite THOL pressure observations to {args.output}")


if __name__ == "__main__":
    main()
