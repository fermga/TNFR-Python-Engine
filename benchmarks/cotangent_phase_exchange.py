"""Bounded C8 execution of the conditional cotangent exchange law.

The momentum premise and pressure correction belong to variational section
13.21. This instrument never changes engine defaults. Callers must supply a
frozen amplitude, Euler clock and step count before evaluating a response.
The retained trajectory is finite numerical evidence, not a formation result
or an integration-error certificate. No target trajectory runs on import.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from tnfr.dynamics._euler_kernel import euler_update  # noqa: E402
from tnfr.alias import get_attr, set_attr  # noqa: E402
from tnfr.constants.aliases import (  # noqa: E402
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.dynamics.fused_dnfr import compute_fused_gradients_symmetric  # noqa: E402
from tnfr.research.core_manifests import current_git_source_provenance  # noqa: E402
from tnfr.sdk import Network, diagnose_network  # noqa: E402


SIZE = 8
EPI_WEIGHT = PHASE_WEIGHT = 0.5
_NODES = np.arange(SIZE)
_PREVIOUS = (_NODES - 1) % SIZE
_NEXT = (_NODES + 1) % SIZE
_SOURCE = np.repeat(_NODES, 2)
_TARGET = np.column_stack((_PREVIOUS, _NEXT)).ravel()
_CAPACITY = np.ones(SIZE)
_WEIGHTS = {"w_epi": EPI_WEIGHT, "w_phase": PHASE_WEIGHT, "w_vf": 0, "w_topo": 0}


@dataclass(frozen=True)
class CotangentCycleSpec:
    """Explicit finite preparation; coefficients and graph are fixed by scope."""

    amplitude: float
    timestep: float
    steps: int
    checkpoints: tuple[int, ...] = ()

    def __post_init__(self):
        for name in ("amplitude", "timestep"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive binary64 float")
        if self.amplitude >= 0.5:
            raise ValueError("the frozen preparation must satisfy amplitude < 1/2")
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(self.steps * self.timestep):
            raise ValueError("the final clock must be representable")
        if type(self.checkpoints) is not tuple or any(
            type(step) is not int or not 0 <= step <= self.steps
            for step in self.checkpoints
        ):
            raise ValueError("checkpoints must be integer steps within the run")


def _state_vector(value, label):
    result = np.asarray(value)
    if result.shape != (SIZE,) or result.dtype.kind not in "fi":
        raise ValueError(f"{label} must be a finite real length-eight vector")
    result = result.astype(float, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def cycle_geometry(phase):
    """H and its full derivative on the declared real lifted acute C8 chart.

    H_i=2*pi*cos(b_i)*sinc(a_i), with a the neighbor-midpoint displacement
    and b half the neighbor separation. The derivative includes the center
    and both neighboring phase coordinates. A series avoids cancellation in
    sinc' near zero; its threshold is a numerical evaluation choice only.
    """
    phase = _state_vector(phase, "phase")
    edges = phase[_NEXT] - phase
    if np.max(np.abs(edges)) >= math.pi / 2:
        raise ValueError("phase leaves the declared strict acute real-lift chart")
    a = (phase[_PREVIOUS] + phase[_NEXT]) / 2 - phase
    b = (phase[_NEXT] - phase[_PREVIOUS]) / 2
    sinc = np.sinc(a / math.pi)
    derivative = np.empty(SIZE)
    small = np.abs(a) <= 0.01
    square = a[small] ** 2
    derivative[small] = a[small] * (
        -1 / 3
        + square
        * (1 / 30 + square * (-1 / 840 + square * (1 / 45360 - square / 3991680)))
    )
    regular = a[~small]
    derivative[~small] = (regular * np.cos(regular) - np.sin(regular)) / regular**2
    cosine, sine = np.cos(b), np.sin(b)
    metric = 2 * math.pi * cosine * sinc
    differential = np.zeros((SIZE, SIZE))
    differential[_NODES, _NODES] = -2 * math.pi * cosine * derivative
    differential[_NODES, _PREVIOUS] = math.pi * (cosine * derivative + sine * sinc)
    differential[_NODES, _NEXT] = math.pi * (cosine * derivative - sine * sinc)
    if not np.all(np.isfinite(metric)) or np.min(metric) <= 0:
        raise ValueError("the cotangent metric must be represented positive")
    return metric, differential, a, edges


def cotangent_cycle_field(form, phase):
    """One detached joint field, using the existing old-pressure array owner."""
    form, phase = _state_vector(form, "form"), _state_vector(phase, "phase")
    metric, differential, displacement, edges = cycle_geometry(phase)
    old_pressure = compute_fused_gradients_symmetric(
        edge_src=_SOURCE,
        edge_dst=_TARGET,
        phase=phase,
        epi=form,
        vf=_CAPACITY,
        weights=_WEIGHTS,
        accumulate_both_directions=False,
        use_jit=False,
    )
    phase_rate = form / metric
    correction = (differential.T @ (form * phase_rate)) / metric - phase_rate * (
        differential @ phase_rate
    )
    pressure = old_pressure + correction
    laplacian_form = form - (form[_PREVIOUS] + form[_NEXT]) / 2
    reference_old = -EPI_WEIGHT * laplacian_form + PHASE_WEIGHT * displacement / math.pi
    gradient = np.sin(edges[_PREVIOUS]) - np.sin(edges)
    dissipation = EPI_WEIGHT * float(form @ laplacian_form)
    energy_rate = float(form @ pressure + PHASE_WEIGHT * gradient @ phase_rate)
    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(phase_rate)):
        raise ValueError("joint field arithmetic must remain representable")
    return {
        "form_rate": pressure,
        "phase_rate": phase_rate,
        "old_pressure": old_pressure,
        "geometric_pressure": correction,
        "metric": metric,
        "metric_differential": differential,
        "edge_gaps": edges,
        "energy": float(
            form @ form / 2 + PHASE_WEIGHT * np.sum(2 * np.sin(edges / 2) ** 2)
        ),
        "dissipation": dissipation,
        "energy_rate_residual": energy_rate + dissipation,
        "geometric_work_residual": float(form @ correction),
        "old_pressure_formula_residual_inf": float(
            np.max(np.abs(old_pressure - reference_old))
        ),
    }


def _checkpoint_graph(form, phase):
    graph = nx.cycle_graph(SIZE)
    graph.graph.update(
        DNFR_WEIGHTS={"epi": EPI_WEIGHT, "phase": PHASE_WEIGHT, "vf": 0, "topo": 0},
        vectorized_dnfr=True,
    )
    for index in graph:
        set_attr(graph.nodes[index], ALIAS_EPI, float(form[index]))
        set_attr(graph.nodes[index], ALIAS_THETA, float(phase[index]))
        set_attr(graph.nodes[index], ALIAS_VF, 1.0)
    default_compute_delta_nfr(graph)
    return graph


def run_cotangent_cycle(spec: CotangentCycleSpec):
    """Execute exactly one declared preparation without tuning or retries."""
    if not isinstance(spec, CotangentCycleSpec):
        raise TypeError("spec must be a validated CotangentCycleSpec")
    mode = np.cos(2 * math.pi * _NODES / SIZE)
    form, phase = spec.amplitude * mode, np.zeros(SIZE)
    checkpoints = {0, spec.steps, *spec.checkpoints}
    samples = []
    maxima = dict.fromkeys(
        (
            "edge_gap_abs",
            "form_abs",
            "phase_abs",
            "energy",
            "energy_norm",
            "geometric_pressure_abs",
            "energy_rate_residual_abs",
            "geometric_work_residual_abs",
            "old_pressure_formula_residual_inf",
            "nodal_increment_residual_inf",
            "phase_increment_residual_inf",
            "positive_energy_increment",
            "euler_energy_balance_defect_abs",
        ),
        0.0,
    )
    metric_min = math.inf
    accumulated_dissipation = 0.0
    initial_energy = None
    previous_energy = None
    source_paths = ("benchmarks/cotangent_phase_exchange.py", "src/tnfr")
    sha, dirty, digest = current_git_source_provenance(ROOT, source_paths)
    for step in range(spec.steps + 1):
        field = cotangent_cycle_field(form, phase)
        energy = field["energy"]
        if initial_energy is None:
            initial_energy = energy
        metric_min = min(metric_min, float(np.min(field["metric"])))
        values = {
            "edge_gap_abs": np.max(np.abs(field["edge_gaps"])),
            "form_abs": np.max(np.abs(form)),
            "phase_abs": np.max(np.abs(phase)),
            "energy": energy,
            "energy_norm": math.sqrt(
                float(form @ form + field["edge_gaps"] @ field["edge_gaps"] / 2)
            ),
            "geometric_pressure_abs": np.max(np.abs(field["geometric_pressure"])),
            "energy_rate_residual_abs": abs(field["energy_rate_residual"]),
            "geometric_work_residual_abs": abs(field["geometric_work_residual"]),
            "old_pressure_formula_residual_inf": field[
                "old_pressure_formula_residual_inf"
            ],
        }
        if previous_energy is not None:
            values["positive_energy_increment"] = energy - previous_energy
        for name, value in values.items():
            maxima[name] = max(maxima[name], float(value))
        maxima["euler_energy_balance_defect_abs"] = max(
            maxima["euler_energy_balance_defect_abs"],
            abs(energy - initial_energy + accumulated_dissipation),
        )
        if step in checkpoints:
            graph = _checkpoint_graph(form, phase)
            graph_pressure = np.array(
                [get_attr(graph.nodes[index], ALIAS_DNFR) for index in graph]
            )
            # The detached SDK observer must see the candidate's pressure,
            # including K*x. Its dEPI is a declared model rate, not measured
            # temporal evidence or a Mutation trigger history.
            for index in graph:
                rate = float(field["form_rate"][index])
                set_attr(graph.nodes[index], ALIAS_DNFR, rate)
                set_attr(graph.nodes[index], ALIAS_DEPI, rate)
            diagnostics = diagnose_network(
                Network(graph, name="conditional_cotangent_c8")
            )
            samples.append(
                {
                    "step": step,
                    "time": step * spec.timestep,
                    "form": form.tolist(),
                    "phase_lift": phase.tolist(),
                    "mode_amplitude": float(mode @ form / 4),
                    "form_mean": float(np.mean(form)),
                    "phase_mean": float(np.mean(phase)),
                    "energy": energy,
                    "energy_norm": values["energy_norm"],
                    "canonical_momentum": float(field["metric"] @ form),
                    "old_pressure": field["old_pressure"].tolist(),
                    "geometric_pressure": field["geometric_pressure"].tolist(),
                    "phase_rate": field["phase_rate"].tolist(),
                    "default_graph_pressure_residual_inf": float(
                        np.max(np.abs(graph_pressure - field["old_pressure"]))
                    ),
                    "diagnostics": diagnostics,
                    "diagnostics_pressure_provenance": (
                        "Refreshed old production pressure plus geometric K*x; "
                        "stored dEPI is the same candidate model rate at held nu=1. "
                        "No retrospective pressure reconstruction or controller."
                    ),
                }
            )
        if step == spec.steps:
            break
        # Both rows use the same immutable input state. Capacity is exactly one;
        # no history, operator event, source registry, clipping or projection.
        next_form = euler_update(form, spec.timestep, field["form_rate"])
        next_phase = euler_update(phase, spec.timestep, field["phase_rate"])
        for name, before, after, rate in (
            ("nodal_increment_residual_inf", form, next_form, field["form_rate"]),
            ("phase_increment_residual_inf", phase, next_phase, field["phase_rate"]),
        ):
            maxima[name] = max(
                maxima[name],
                float(np.max(np.abs((after - before) - spec.timestep * rate))),
            )
        accumulated_dissipation += spec.timestep * field["dissipation"]
        previous_energy = energy
        form, phase = next_form, next_phase
    return {
        "scope": "Finite conditional C8 cotangent-law execution; not an installed engine default or a validated ODE enclosure.",
        "specification": asdict(spec),
        "model": {
            "graph": "unit-conductance reciprocal C8",
            "node_order": list(range(SIZE)),
            "capacity": 1,
            "epi_weight": EPI_WEIGHT,
            "phase_weight": PHASE_WEIGHT,
            "form_row": "old_pressure + K*x",
            "phase_row": "inverse(H)*x",
            "initial_form": "amplitude*cos(2*pi*i/8)",
            "initial_phase": 0,
            "forcing": "none",
            "operators": [],
            "seed": None,
            "clock": "step_index*timestep; no accumulated floating clock",
            "final_time": spec.steps * spec.timestep,
        },
        "execution": {
            "old_pressure_owner": "tnfr.dynamics.fused_dnfr.compute_fused_gradients_symmetric(use_jit=False)",
            "checkpoint_pressure_owner": "tnfr.dynamics.dnfr.default_compute_delta_nfr",
            "nodal_update_owner": "tnfr.dynamics._euler_kernel.euler_update",
            "scheme": "simultaneous explicit Euler, held unit capacity",
            "phase_chart": "real lift; every retained state has strict acute adjacent gaps",
            "energy_norm": "sqrt(x dot x + sum(edge_gaps squared)/2)",
            "state_retention": "All step endpoints checked; selected endpoints retained as complete eight-node x/phase states with fixed capacity/support.",
            "residual_scope": "All-step numerical maxima; no validated uniform floating-point error bound or continuous-ODE enclosure.",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "precision": "binary64",
            "git_sha": sha,
            "scoped_dirty": dirty,
            "working_source_digest": digest,
            "source_paths": source_paths,
        },
        "samples": samples,
        "all_step_maxima": maxima,
        "minimum_metric": metric_min,
        "accumulated_euler_dissipation": accumulated_dissipation,
        "final_energy_balance_defect": energy
        - initial_energy
        + accumulated_dissipation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", required=True, type=float)
    parser.add_argument("--timestep", required=True, type=float)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--checkpoint", action="append", type=int, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_cotangent_cycle(
        CotangentCycleSpec(
            args.amplitude,
            args.timestep,
            args.steps,
            tuple(args.checkpoint),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved conditional finite execution to {args.output}")


if __name__ == "__main__":
    main()
