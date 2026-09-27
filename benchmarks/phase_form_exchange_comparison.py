"""Frozen finite constitutive probes for two conditional phase/form laws.

These are prepared tangent-line states, not evolved temporal endpoints. The
native pressure owner evaluates their form response. On each pure mode the
cotangent correction has exactly zero projection by skewness; its complete
vector is not computed or silently identified with native pressure.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import platform

import networkx as nx
import numpy as np

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics.canonical import compute_canonical_nodal_derivative
from tnfr.dynamics.dnfr import default_compute_delta_nfr

ROOT = Path(__file__).resolve().parents[1]
AMPLITUDE, PROBE_SPAN = Q(1, 16), Q(1, 8)
EPI_WEIGHT = PHASE_WEIGHT = Q(1, 2)
BETA, ETA = Q(1, 2), Q(1)
PI_REPRESENTED = Q(math.pi)
Q_ALLOWANCE = Q(1, 10**10)
PRESSURE_PATH = "fused_canonical"
MODES = {
    "low": (Q(1, 2), (2, 1, -1, -2, -1, 1)),
    "high": (Q(3, 2), (2, -1, -1, 2, -1, -1)),
}
FINGERPRINT_PATHS = (
    "benchmarks/phase_form_exchange_comparison.py",
    "src/tnfr/dynamics/dnfr.py",
    "src/tnfr/dynamics/fused_dnfr.py",
    "src/tnfr/dynamics/canonical.py",
    "src/tnfr/config/defaults_core.py",
    "src/tnfr/_exact_time.py",
    "src/tnfr/mathematics/_neighbor_differences.py",
    "src/tnfr/mathematics/_phase_midpoint.py",
    "src/tnfr/mathematics/unified_numerical.py",
)


def _laplacian(values):
    return tuple(
        x - (values[i - 1] + values[(i + 1) % 6]) / 2 for i, x in enumerate(values)
    )


def _projection(values, mode):
    return sum((x * v for x, v in zip(values, mode, strict=True)), Q(0)) / sum(
        v * v for v in mode
    )


def _restoring(responses, eigenvalue, mode):
    acceleration_over_amplitude = (
        _projection(responses["plus"], mode) - _projection(responses["minus"], mode)
    ) / (2 * PROBE_SPAN * AMPLITUDE)
    return EPI_WEIGHT**2 * eigenvalue**2 - acceleration_over_amplitude


def _ratio_interval(low, high):
    if low <= Q_ALLOWANCE:
        raise ValueError("restoring denominator is not separated from its error budget")
    return (
        (high - Q_ALLOWANCE) / (low + Q_ALLOWANCE),
        (high + Q_ALLOWANCE) / (low - Q_ALLOWANCE),
    )


def prepare_prediction():
    """Derive probes, exact represented references and decisions before pressure."""
    cases, ratios = {}, {}
    for model in ("relational", "cotangent"):
        cases[model] = {}
        for label, (eigenvalue, mode) in MODES.items():
            if tuple(_laplacian(tuple(map(Q, mode)))) != tuple(
                eigenvalue * v for v in mode
            ):
                raise ValueError("preparation is not the declared pure C6 mode")
            gain = (
                PHASE_WEIGHT * eigenvalue / (BETA * PI_REPRESENTED)
                if model == "relational"
                else 1 / (ETA * PI_REPRESENTED * 2)
            )
            probes, reference = {}, {}
            for name, sign in (("minus", -1), ("plus", 1)):
                form = tuple(
                    AMPLITUDE * (1 - sign * PROBE_SPAN * EPI_WEIGHT * eigenvalue) * v
                    for v in mode
                )
                phase = tuple(
                    Q(float(sign * PROBE_SPAN * gain * AMPLITUDE * v)) for v in mode
                )
                if max(phase) - min(phase) >= Q(1, 4):
                    raise ValueError("probe leaves the frozen acute midpoint domain")
                reference[name] = tuple(
                    -EPI_WEIGHT * x - PHASE_WEIGHT * theta / PI_REPRESENTED
                    for x, theta in zip(
                        _laplacian(form), _laplacian(phase), strict=True
                    )
                )
                probes[name] = {"form": form, "phase": phase}
            restoring = _restoring(reference, eigenvalue, mode)
            cases[model][label] = {
                "eigenvalue": eigenvalue,
                "mode": mode,
                "phase_gain": gain,
                "probes": probes,
                "reference_rate": reference,
                "reference_restoring": restoring,
            }
        low = cases[model]["low"]["reference_restoring"]
        high = cases[model]["high"]["reference_restoring"]
        ratios[model] = {
            "exact_real_prediction": 9 if model == "relational" else 3,
            "represented_reference": high / low,
            "admitted_interval": _ratio_interval(low, high),
        }
    if (
        not ratios["cotangent"]["admitted_interval"][1]
        < 6
        < ratios["relational"]["admitted_interval"][0]
    ):
        raise ValueError("the prospective ratio intervals do not discriminate")
    return {
        "protocol": "phase-form-exchange-finite-probe-v1",
        "scope": "prepared_constitutive_states_not_executed_trajectory_or_physical_selection",
        "support": "unit_undirected_C6_in_node_order_0_to_5",
        "capacity": 1,
        "amplitude": AMPLITUDE,
        "probe_span": PROBE_SPAN,
        "epi_weight": EPI_WEIGHT,
        "phase_weight": PHASE_WEIGHT,
        "beta": BETA,
        "eta": ETA,
        "scale_choice": "a_priori_low_mode_match_beta_equals_eta_degree_w_lambda_low",
        "pi_represented": PI_REPRESENTED,
        "restoring_allowance": Q_ALLOWANCE,
        "cases": cases,
        "ratios": ratios,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "precision": "binary64",
            "pressure_path": PRESSURE_PATH,
            "kernel": "numpy_fused_unit_edges_below_jit_size_threshold",
            "randomness": "none",
        },
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in FINGERPRINT_PATHS
        },
    }


def _native_rate(probe):
    graph = nx.cycle_graph(6)
    graph.graph["vectorized_dnfr"] = True
    graph.graph["DNFR_WEIGHTS"] = {"epi": 0.5, "phase": 0.5, "vf": 0.0, "topo": 0.0}
    for i in graph:
        graph.nodes[i].update(
            EPI=float(probe["form"][i]), theta=float(probe["phase"][i]), nu_f=1.0
        )
    profile = {}
    default_compute_delta_nfr(graph, profile=profile)
    if profile.get("dnfr_path") != PRESSURE_PATH:
        raise ValueError("pressure execution differs from the frozen path")
    return tuple(
        Q(
            compute_canonical_nodal_derivative(
                1.0, get_attr(graph.nodes[i], ALIAS_DNFR, strict=True)
            ).derivative
        )
        for i in graph
    )


def evaluate_prediction(prediction):
    """Observe native pressure, admitting each defect before any ratio verdict."""
    if prediction != prepare_prediction():
        raise ValueError("prediction differs from the frozen protocol/source")
    observed = {}
    for model, cases in prediction["cases"].items():
        rows = {}
        for label, case in cases.items():
            responses = {
                name: _native_rate(probe) for name, probe in case["probes"].items()
            }
            defects = {
                name: tuple(
                    a - b for a, b in zip(responses[name], reference, strict=True)
                )
                for name, reference in case["reference_rate"].items()
            }
            restoring = _restoring(responses, case["eigenvalue"], case["mode"])
            # Exact arithmetic on observed represented rates; propagated
            # endpoint kernel defects, not a time-integration error estimate.
            budget = sum(
                abs(_projection(defect, case["mode"])) for defect in defects.values()
            ) / (2 * PROBE_SPAN * AMPLITUDE)
            if budget > Q_ALLOWANCE:
                raise ValueError(
                    "pressure realization exceeds the frozen restoring allowance"
                )
            if abs(restoring - case["reference_restoring"]) > budget:
                raise ValueError("restoring response exceeds its measured defect bound")
            rows[label] = {
                "native_rate": responses,
                "rate_defect": defects,
                "restoring": restoring,
                "restoring_error_bound": budget,
            }
        ratio = rows["high"]["restoring"] / rows["low"]["restoring"]
        lower, upper = prediction["ratios"][model]["admitted_interval"]
        if not lower <= ratio <= upper:
            raise ValueError("observed ratio is outside its prospective interval")
        observed[model] = {"modes": rows, "ratio": ratio}
    return {
        "prediction": prediction,
        "observed": observed,
        "passed": observed["cotangent"]["ratio"] < 6 < observed["relational"]["ratio"],
        "cotangent_projection": "Kx_omitted_only_after_exact_skew_projection_on_collinear_form; full_vector_not_evaluated",
        "scope": "implementation_consistency_of_two_declared_laws_not_model_selection_or_temporal_data",
    }


def _encoded(value):
    def exact(item):
        if isinstance(item, Q):
            return {"numerator": item.numerator, "denominator": item.denominator}
        raise TypeError(f"unsupported report value {type(item).__name__}")

    return json.dumps(value, default=exact, indent=2, sort_keys=True) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prediction = prepare_prediction()
    path = args.output.with_suffix(".prediction.json")
    if args.prepare:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_encoded(prediction))
        print(f"Prediction frozen: {path}")
        return 0
    if path.read_text(encoding="utf-8") != _encoded(prediction):
        raise ValueError("freeze the current protocol before evaluation")
    if args.output.exists():
        raise FileExistsError("refusing to replace a retained response")
    try:
        report = evaluate_prediction(prediction)
    except Exception as error:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(
                _encoded(
                    {"prediction": prediction, "passed": False, "error": repr(error)}
                )
            )
        raise
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_encoded(report))
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "ratios": {
                    name: float(row["ratio"])
                    for name, row in report["observed"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
