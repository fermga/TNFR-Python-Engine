"""Frozen F4 response of two source-relative form preparations.

This bounded workstation control evaluates a held S0 law through the existing
pressure kernel and unforced integrator. It selects neither a new pressure law
nor an autonomous source. Run ``--prepare`` before the first evaluation to
retain the prediction and source fingerprints independently of the response.
The current fail-closed runner reproduces an already evaluated scientific
protocol; the original producer and reserved records are retained separately.
"""

from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import math
import platform
from fractions import Fraction as Q
from pathlib import Path

import networkx as nx
import numpy as np

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.physics.source_relative_form import observe_source_relative_form

NODES = tuple(range(6))
REGIONS = ((0, 1, 2), (3, 4, 5))
DT, STEPS = Q(1, 128), 128
IDEAL_SOURCE = (Q(1, 12), Q(-1, 24), Q(-1, 24)) * 2
INITIAL = {
    "plus": (Q(1, 2), Q(5, 8), Q(3, 8)) * 2,
    "minus": (Q(1, 2), Q(3, 8), Q(5, 8)) * 2,
}
GENERATOR = tuple(
    tuple(
        (
            Q(-1, 2)
            if j == i
            else (Q(1, 4) if j in (3 * (i // 3) + (i + 1) % 3, (i + 3) % 6) else Q(0))
        )
        for j in NODES
    )
    for i in NODES
)
RUNTIME_ALLOWANCE = SOURCE_ALLOWANCE = Q(1, 10**12)
GAP_THRESHOLD = Q(1, 1024)
ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_PATHS = (
    "benchmarks/source_relative_form_response.py",
    "src/tnfr/dynamics/dnfr.py",
    "src/tnfr/dynamics/fused_dnfr.py",
    "src/tnfr/dynamics/integrators.py",
    "src/tnfr/dynamics/_euler_kernel.py",
    "src/tnfr/dynamics/canonical.py",
    "src/tnfr/config/defaults_core.py",
    "src/tnfr/mathematics/_phase_midpoint.py",
    "src/tnfr/physics/form_geometry.py",
    "src/tnfr/physics/source_relative_form.py",
)


def _require(condition, message):
    """Enforce scientific admission even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def _apply(matrix, values):
    return tuple(sum(a * b for a, b in zip(row, values, strict=True)) for row in matrix)


def _rate(values, source):
    return tuple(a + b for a, b in zip(_apply(GENERATOR, values), source, strict=True))


def _intensity(values):
    x, y, z = values[:3]
    return (x - y) ** 2 / 2 + (x + y - 2 * z) ** 2 / 6


def _intensity_error(radius, error):
    return 6 * radius * error + 3 * error**2


def _exact_euler(initial, source):
    values = initial
    for _ in range(STEPS):
        values = tuple(
            x + DT * v for x, v in zip(values, _rate(values, source), strict=True)
        )
    return values


def prepare_prediction():
    """Freeze analytic/reference predictions without evolving the engine."""
    time = DT * STEPS
    radius = Q(1, 8) + time * max(map(abs, IDEAL_SOURCE))
    rows = {}
    a = complex(-3 / 8, -math.sqrt(3) / 8)
    c = complex(1 / (8 * math.sqrt(2)), 1 / (8 * math.sqrt(6)))
    for name, initial in INITIAL.items():
        velocity = _rate(initial, IDEAL_SOURCE)
        truncation = time * DT * max(map(abs, _apply(GENERATOR, velocity))) / 2
        reference = _exact_euler(initial, IDEAL_SOURCE)
        total_error = truncation + RUNTIME_ALLOWANCE + SOURCE_ALLOWANCE
        sign = 1 if name == "plus" else -1
        z = cmath.exp(a * float(time)) * (sign * 1j * math.sqrt(3) * c)
        z += (cmath.exp(a * float(time)) - 1) * c / a
        rows[name] = {
            "initial": initial,
            "initial_rate": velocity,
            "euler_final": reference,
            "euler_intensity": _intensity(reference),
            "truncation_bound": truncation,
            "intensity_error_budget": _intensity_error(radius, total_error),
            "continuous_intensity_estimate": abs(z) ** 2,
        }
    # sin(y)>=y*(1-y²/6), exp(-3/4)>=1/4; no floating exp certifies this cut.
    continuous_gap_lower = Q(127, 65536)
    intensity_budget = sum(row["intensity_error_budget"] for row in rows.values())
    _require(
        continuous_gap_lower - intensity_budget > GAP_THRESHOLD,
        "prospective numerical budget does not separate the response threshold",
    )
    return {
        "protocol": "F4-source-relative-response-v2",
        "model": "held_phase_capacity_support_affine_S0_unforced_scalar_Euler",
        "nodes": NODES,
        "regions": REGIONS,
        "generator": GENERATOR,
        "ideal_source": IDEAL_SOURCE,
        "phase": (0.0, math.pi / 3, math.pi / 6) * 2,
        "pressure_weights": dict(phase=0.5, epi=0.5, vf=0.0, topo=0.0),
        "capacity": (1,) * 6,
        "dt": DT,
        "steps": STEPS,
        "horizon": time,
        "clipping_interval": (0, 1),
        "form_radius_bound": radius,
        "runtime_allowance": RUNTIME_ALLOWANCE,
        "source_allowance": SOURCE_ALLOWANCE,
        "continuous_gap_lower": continuous_gap_lower,
        "observed_gap_threshold": GAP_THRESHOLD,
        "ablation_gap_tolerance": 2 * _intensity_error(Q(1, 8), RUNTIME_ALLOWANCE),
        "trajectories": rows,
        "ablation": "same_weights_and_preparations_uniform_held_phase",
        "reference_scope": "exact_Euler_rationals_and_analytic_bound; complex_exp_is_estimate",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "precision": "binary64",
            "randomness": "none",
        },
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in FINGERPRINT_PATHS
        },
    }


def _graph(initial, *, ablation=False):
    graph = nx.DiGraph()
    graph.add_nodes_from(NODES)
    for node in NODES:
        graph.add_edge(node, 3 * (node // 3) + (node + 1) % 3, weight=1.0)
        graph.add_edge(node, (node + 3) % 6, weight=1.0)
        graph.nodes[node].update(
            EPI=float(initial[node]),
            nu_f=1.0,
            theta=0.0 if ablation else (0.0, math.pi / 3, math.pi / 6)[node % 3],
        )
    graph.graph.update(
        DNFR_WEIGHTS=dict(phase=0.5, epi=0.5, vf=0.0, topo=0.0),
        GAMMA={"type": "none"},
        use_extended_dynamics=False,
        DT_MIN=0.0,
        EPI_MIN=0.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
    )
    return graph


def _read(graph, aliases):
    return tuple(Q(get_attr(graph.nodes[node], aliases, strict=True)) for node in NODES)


def _observation(graph, represented_source):
    relative = observe_source_relative_form(
        graph, REGIONS, held_source_rate=represented_source
    )
    report = relative.form
    return {
        "epi": report.epi,
        "means": tuple(row.mean for row in report.regions),
        "gram_real": report.gram_real,
        "gram_rate_real": report.gram_rate_real,
        "gram_imag_numerator": report.gram_imag_numerator,
        "gram_rate_imag_numerator": report.gram_rate_imag_numerator,
        "intensity": report.regions[0].intensity,
        "intensity_rate": report.regions[0].intensity_rate,
        "nodal_rate_rounding_defect": report.nodal_rate_rounding_defect,
        "relative_real": relative.relative_real,
        "relative_imag_numerator": relative.relative_imag_numerator,
        "relative_rate_real": relative.relative_rate_real,
        "relative_rate_imag_numerator": relative.relative_rate_imag_numerator,
    }


def _execute(name, represented_source, *, ablation=False):
    graph = _graph(INITIAL[name], ablation=ablation)
    ideal_source = (Q(0),) * 6 if ablation else IDEAL_SOURCE
    source_error = max(abs(a - b) for a, b in zip(represented_source, ideal_source))
    _require(
        DT * STEPS * source_error <= SOURCE_ALLOWANCE,
        "prospective source discrepancy exceeds the frozen allowance",
    )
    frozen = (
        _read(graph, ALIAS_THETA),
        _read(graph, ALIAS_VF),
        tuple(graph.edges(data="weight")),
    )
    default_compute_delta_nfr(graph)
    initial_observation = _observation(graph, represented_source)
    reference = INITIAL[name]
    runtime_bound = max_pressure_defect = max_integration_defect = Q(0)
    max_form_error = Q(0)
    frozen_state_preserved = clipping_inactive = True
    for step in range(STEPS):
        before = _read(graph, ALIAS_EPI)
        default_compute_delta_nfr(graph)
        pressure = _read(graph, ALIAS_DNFR)
        modeled = _rate(before, represented_source)
        pressure_defect = tuple(p - v for p, v in zip(pressure, modeled, strict=True))
        update_epi_via_nodal_equation(
            graph, dt=float(DT), t=float(step * DT), method="euler"
        )
        after = _read(graph, ALIAS_EPI)
        integration_defect = tuple(
            new - old - DT * p
            for new, old, p in zip(after, before, pressure, strict=True)
        )
        pd, ed = max(map(abs, pressure_defect)), max(map(abs, integration_defect))
        runtime_bound += DT * pd + ed
        max_pressure_defect = max(max_pressure_defect, pd)
        max_integration_defect = max(max_integration_defect, ed)
        reference = tuple(
            x + DT * v
            for x, v in zip(reference, _rate(reference, ideal_source), strict=True)
        )
        error = max(abs(x - y) for x, y in zip(after, reference, strict=True))
        max_form_error = max(max_form_error, error)
        _require(
            error <= runtime_bound + (step + 1) * DT * source_error,
            f"step {step}: fine-state error exceeds the accumulated defect bound",
        )
        _require(
            runtime_bound <= RUNTIME_ALLOWANCE,
            f"step {step}: arithmetic defects exceed the frozen allowance",
        )
        clipping_inactive = clipping_inactive and min(after) > 0 and max(after) < 1
        _require(clipping_inactive, f"step {step}: boundary projection may have acted")
        frozen_state_preserved = frozen_state_preserved and frozen == (
            _read(graph, ALIAS_THETA),
            _read(graph, ALIAS_VF),
            tuple(graph.edges(data="weight")),
        )
        _require(
            frozen_state_preserved,
            f"step {step}: held phase, capacity or support changed",
        )
    default_compute_delta_nfr(graph)
    final_observation = _observation(graph, represented_source)
    return {
        "initial_observation": initial_observation,
        "final_observation": final_observation,
        "runtime_bound": runtime_bound,
        "source_model_bound": DT * STEPS * source_error,
        "max_pressure_defect": max_pressure_defect,
        "max_integration_defect": max_integration_defect,
        "max_form_error_vs_exact_Euler": max_form_error,
        "final_error_vs_exact_Euler": max(
            abs(x - y) for x, y in zip(final_observation["epi"], reference)
        ),
        "frozen_state_preserved": frozen_state_preserved,
        "clipping_inactive": clipping_inactive,
    }


def evaluate_prediction(prediction):
    """Evaluate the unchanged frozen specification and preserve any failure."""
    if prediction != prepare_prediction():
        raise ValueError("prediction differs from the current frozen protocol/source")
    uniform = _graph((Q(1, 2),) * 6)
    default_compute_delta_nfr(uniform)
    source = _read(uniform, ALIAS_DNFR)
    _require(source[:3] == source[3:], "native source differs between regions")
    _require(
        source[0] == -2 * source[1] and source[1] == source[2],
        "native source does not have the declared contrast orientation",
    )
    _require(source[0] > 0, "native source does not have the declared positive sign")
    # The uniform-form evaluation isolates prospective native pressure; no
    # observed EPI derivative or evaluated trajectory defines the source.
    rows = {name: _execute(name, source) for name in INITIAL}
    zero_graph = _graph((Q(1, 2),) * 6, ablation=True)
    default_compute_delta_nfr(zero_graph)
    zero_source = _read(zero_graph, ALIAS_DNFR)
    _require(zero_source == (0,) * 6, "uniform-phase ablation retains a nonzero source")
    ablation = {name: _execute(name, zero_source, ablation=True) for name in INITIAL}
    gap = (
        rows["plus"]["final_observation"]["intensity"]
        - rows["minus"]["final_observation"]["intensity"]
    )
    ablation_gap = (
        ablation["plus"]["final_observation"]["intensity"]
        - ablation["minus"]["final_observation"]["intensity"]
    )
    initial_rate_defects = {
        name: row["initial_observation"]["intensity_rate"] + Q(3, 128)
        for name, row in rows.items()
    }
    _require(
        all(
            row["initial_observation"]["means"] == (Q(1, 2),) * 2
            for row in rows.values()
        ),
        "initial regional means differ from the frozen preparation",
    )
    _require(
        rows["plus"]["initial_observation"]["gram_real"]
        == rows["minus"]["initial_observation"]["gram_real"],
        "reflected preparations do not have the same initial Gram matrix",
    )
    _require(
        max(map(abs, initial_rate_defects.values())) <= RUNTIME_ALLOWANCE,
        "initial intensity rates exceed the frozen allowance",
    )
    # A repeated observation of H and Hdot is not assigned a fresh identity
    # after evaluation. Report its actual represented discrepancy explicitly.
    return {
        "prediction": prediction,
        "source": source,
        "trajectories": rows,
        "ablation": ablation,
        "observed_gap": gap,
        "ablation_gap": ablation_gap,
        "initial_intensity_rate_defects": initial_rate_defects,
        "passed": gap > GAP_THRESHOLD
        and abs(ablation_gap) <= prediction["ablation_gap_tolerance"],
        "scope": "finite_internal_information_test_not_physical_validation_or_formation",
    }


def _json_default(value):
    if isinstance(value, Q):
        return {"numerator": value.numerator, "denominator": value.denominator}
    raise TypeError(f"unsupported report value {type(value).__name__}")


def _encoded(value):
    return json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prediction = prepare_prediction()
    frozen_path = args.output.with_suffix(".prediction.json")
    if args.prepare:
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        if frozen_path.exists() and frozen_path.read_text(encoding="utf-8") != _encoded(
            prediction
        ):
            raise ValueError("refusing to overwrite a different frozen prediction")
        frozen_path.write_text(_encoded(prediction), encoding="utf-8")
        print(f"Prediction frozen: {frozen_path}")
        return 0
    if frozen_path.read_text(encoding="utf-8") != _encoded(prediction):
        raise ValueError("freeze the current protocol before evaluation")
    if args.output.exists():
        raise FileExistsError(
            "refusing to overwrite an evaluated response; use a new path"
        )
    try:
        report = evaluate_prediction(prediction)
    except Exception as error:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(
                _encoded(
                    {"prediction": prediction, "passed": False, "error": repr(error)}
                )
            )
        raise
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(_encoded(report))
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "gap": float(report["observed_gap"]),
                "threshold": float(GAP_THRESHOLD),
                "ablation_gap": float(report["ablation_gap"]),
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
