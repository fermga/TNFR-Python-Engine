"""Reproduce P4/P5 memory references and one declared REMESH runtime witness.

Run from the repository root with ``python benchmarks/derived_epi_memory.py``.
The JSON separates exact references, finite numerical observations and the
direct runtime map. Proofs live in theory/DERIVED_EPI_MEMORY.md.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import fields, is_dataclass, replace
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from tnfr.alias import set_attr  # noqa: E402
from tnfr._exact_time import fraction_upper_signed_float  # noqa: E402
from tnfr.constants.aliases import ALIAS_VF  # noqa: E402
from tnfr.operators import apply_network_remesh  # noqa: E402
from tnfr.physics.epi_memory import observe_epi_memory  # noqa: E402
from tnfr.physics.p5_memory_truncation import (  # noqa: E402
    bound_p5_memory_truncation,
)
from tnfr.physics.p5_reduction import (  # noqa: E402
    observe_p5_remesh_reduction,
    reduce_p5_state,
)
from tnfr.research.claims import ClaimStatus  # noqa: E402
from tnfr.research.core_manifests import (  # noqa: E402
    CoreExperimentManifest,
    current_git_source_provenance,
)


def _case(size, partition, initial, times):
    graph = nx.path_graph(size)
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_VF, 1.0)
    result = observe_epi_memory(graph, partition, initial, times=times)
    samples = []
    for sample in result.samples:
        row = {}
        for field in fields(sample):
            value = getattr(sample, field.name)
            row[field.name] = value.tolist() if isinstance(value, np.ndarray) else value
        row["markov_omission_error_inf"] = float(
            np.max(np.abs(sample.projected_epi - sample.markov_epi))
        )
        row["initial_source_omission_error_inf"] = float(
            np.max(np.abs(sample.projected_epi - sample.source_free_epi))
        )
        samples.append(row)
    return {
        "graph": f"unit-weight path P{size}, node order 0..{size - 1}",
        "partition": partition,
        "initial_epi": initial,
        "projection": result.projection.tolist(),
        "micro_generator": result.micro_generator.tolist(),
        "instantaneous_generator": result.instantaneous_generator.tolist(),
        "quotient_generator": result.quotient_generator.tolist(),
        "macro_metric_weights": result.macro_metric_weights.tolist(),
        "closure_within_tolerance": result.closure_within_tolerance,
        "tolerance": result.tolerance,
        "numerical_tolerance": result.numerical_tolerance,
        "generator_residual": result.generator_residual,
        "right_inverse_residual": result.right_inverse_residual,
        "projector_residual": result.projector_residual,
        "quotient_generator_residual": result.quotient_generator_residual,
        "samples": samples,
        "scope": result.scope,
    }


def _exact_payload(value):
    """Preserve rational endpoints; JSON decimal floats are not exact bounds."""
    if isinstance(value, Fraction):
        return str(value)
    if is_dataclass(value):
        return {
            field.name: _exact_payload(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_exact_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _exact_payload(item) for key, item in value.items()}
    return value


def _truncation_case(initial, window, times):
    result = bound_p5_memory_truncation(
        initial, memory_window=window, times=times
    )
    payload = _exact_payload(result)
    for row, sample in zip(payload["samples"], result.samples, strict=True):
        low, high = sample.macro_error_enclosure
        row["macro_error_binary64_interval"] = [
            -fraction_upper_signed_float(-low),
            fraction_upper_signed_float(high),
        ]
        row["macro_error_bound_upper_binary64"] = fraction_upper_signed_float(
            sample.macro_error_bound
        )
        row["evaluation_width_upper_binary64"] = fraction_upper_signed_float(
            sample.evaluation_width
        )
    return payload


def _reduction_case(history):
    result = observe_p5_remesh_reduction(history, alpha=Fraction(1, 2))
    names = (
        "exact_next_field", "exact_next_energy", "exact_history_energies",
        "exact_augmented_energy_before", "exact_augmented_energy_after",
        "exact_energy_drop",
    )
    return _exact_payload({
        "history_newest_first": tuple(tuple(row) for row in history),
        "geometry": result.geometry,
        "alpha": result.certificate.alpha,
        "delays": (result.certificate.tau_local, result.certificate.tau_global),
        "fine": {name: getattr(result.fine, name) for name in names},
        "orbit": {name: getattr(result.orbit, name) for name in names},
        "discarded": {name: getattr(result.discarded, name) for name in names},
        "commutation_residual": result.commutation_residual,
        "energy_split_residual": result.augmented_energy_split_residual,
    })


def _runtime_causal_echo(history):
    """One production map with analytically realizable, caller-supplied history."""
    graph = nx.path_graph(5)
    for node, value in enumerate(history[0]):
        graph.nodes[node].update(EPI=float(value), nu_f=1.0, theta=0.0)
    graph.graph.update(
        REMESH_ALPHA_HARD=True, REMESH_ALPHA=0.5,
        REMESH_TAU_LOCAL=1, REMESH_TAU_GLOBAL=2,
        EPI_MIN=0.0, EPI_MAX=8.0, CLIP_MODE="hard", REMESH_LOG_EVENTS=False,
        _epi_hist=deque(
            [dict(enumerate(row)) for row in reversed(history)], maxlen=8
        ),
    )
    result = apply_network_remesh(
        graph, include_stability_evidence=True, metric_weights=(1, 2, 2, 2, 1)
    )
    output = tuple(Fraction.from_float(p.bounded_epi) for p in result.proposals)
    return _exact_payload({
        "output_epi": output,
        "output_orbits": reduce_p5_state(output).orbit_epi,
        "observed_mode_amplitude": (output[0] - output[2]) / 2,
        "clipping_intervened": result.plan.any_clipping_intervention,
        "rounding_residual_max": result.evidence.max_raw_affine_rounding_residual,
        "scope": "One direct map; supplied history is not executor provenance",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/research/derived_epi_memory.json",
    )
    args = parser.parse_args()
    source_scope = ("src/tnfr", "benchmarks/derived_epi_memory.py")
    sha, dirty, digest = current_git_source_provenance(ROOT, source_scope)
    times = [0.0, 0.125, 0.5, 1.0, 2.0]
    manifest = CoreExperimentManifest(
        claim_id="O4.a-B2-fixed-EPI-projected-memory",
        git_sha=sha,
        source_dirty=dirty,
        dirty_source_hash=digest,
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
        },
        graph_construction="Deterministic unit-weight P4 and P5; no random seed",
        capacity_specification="nu_f=1 at every node, fixed throughout",
        solver="Offline shared matrix exponential; no runtime solver or timestep",
        result_status=ClaimStatus.MEASURED,
        telemetry=(
            "kernel and hidden-state source", "memory convolution",
            "projected and omission-control trajectories", "rate identity residual",
            "generator and stochastic semigroup residuals",
        ),
        controls=(
            "P4 exact quotient", "P5 same macrostate with different hidden state",
            "P5 initially zero hidden source with subsequently induced memory",
        ),
        artifacts=(str(args.output),),
    )
    manifest.validate_for_admission()
    truncation_manifest = replace(
        manifest,
        claim_id="O4.a-B2.b-P5-finite-history-error",
        graph_construction="Abstract unit-weight P5, partition ((0,4),(1,2,3))",
        solver="Exact finite method of steps with rational exponential enclosures",
        result_status=ClaimStatus.DERIVED,
        telemetry=(
            "exact rational full/truncated trajectory intervals",
            "separate omitted forcing and trajectory-error bounds",
            "evaluation enclosure widths; outward-rounded binary64 summaries",
        ),
        controls=(
            "zero history retains initial hidden source",
            "window covering the horizon is exact",
            "three initial states; common capacity fixed at one",
        ),
    )
    truncation_manifest.validate_for_admission()
    p5_partition = [[0, 4], [1, 2, 3]]
    report = {
        "manifest": manifest.to_dict(),
        "source_scope": source_scope,
        "times": times,
        "cases": {
            "p4_closed": _case(4, [[0, 3], [1, 2]], [2, -1, 2, 0], times),
            "p5_equal_middle": _case(5, p5_partition, [0, 1, 1, 1, 0], times),
            "p5_hidden_contrast": _case(5, p5_partition, [0, 0, 3, 0, 0], times),
            "p5_zero_initial_hidden": _case(5, p5_partition, [1, 0, 0, 0, 1], times),
        },
        "empirical_status": "Untested: no physical data or experiment",
        "proof": "theory/DERIVED_EPI_MEMORY.md",
    }
    report["p5_memory_truncation"] = {
        "manifest": truncation_manifest.to_dict(),
        "exact_encoding": "Rational p/q strings are authoritative endpoints",
        "cases": {
            f"{name}_window_{window}": _truncation_case(
                case["initial_epi"], window, times
            )
            for name, case in report["cases"].items()
            if name.startswith("p5_")
            for window in (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(2))
        },
    }
    reduction_manifest = replace(
        truncation_manifest,
        claim_id="O4.a-B2.c-P5-orbit-REMESH-compatibility",
        graph_construction="P5 reflection quotient ((0,4),(1,3),(2,)) to P3",
        solver="Exact rational projection and existing REMESH history theorem",
        telemetry=(
            "generator intertwining and minimal linear dimension",
            "fine/orbit/discarded augmented energy split", "REMESH commutation",
        ),
        controls=(
            "causal decaying-mode echo obstruction",
            "invisible antisymmetric state", "arbitrary asymmetric history",
        ),
    )
    reduction_manifest.validate_for_admission()
    eigenmode = (1, 0, -1, 0, 1)
    causal = tuple(tuple(4 + scale * v for v in eigenmode) for scale in (1, 2, 4))
    hidden = ((1, 2, 0, -2, -1),) * 3
    asymmetric = ((1, 2, 3, -1, 0), (2, -1, 0, 4, 3), (0, 1, -2, 1, 2))
    runtime_manifest = replace(
        reduction_manifest,
        claim_id="O4.a-B2.c-P5-direct-runtime-echo-witness",
        solver="One production apply_network_remesh on a caller-prepared P5",
        result_status=ClaimStatus.MEASURED,
        telemetry=("captured endpoint", "rounding and clipping interventions"),
    )
    runtime_manifest.validate_for_admission()
    report["p5_remesh_reduction"] = {
        "manifest": reduction_manifest.to_dict(),
        "cases": {
            "causal_decay": _reduction_case(causal),
            "discarded_odd": _reduction_case(hidden),
            "asymmetric": _reduction_case(asymmetric),
        },
        "causal_decay_obstruction": {
            "source_sampling": "nu=1, h=log(2), current/local/global times 0,-h,-2h",
            "forward_amplitude": "1/2", "echo_amplitude": "11/4",
            "runtime_manifest": runtime_manifest.to_dict(),
            "runtime": _runtime_causal_echo(causal),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")
    for name, case in report["cases"].items():
        error = max(row["markov_omission_error_inf"] for row in case["samples"])
        residual = max(row["identity_residual"] for row in case["samples"])
        print(f"{name}: max omission={error:.6g}; rate residual={residual:.3g}")
    for name, case in report["p5_memory_truncation"]["cases"].items():
        error = max(row["macro_error_binary64_interval"][1] for row in case["samples"])
        bound = max(row["macro_error_bound_upper_binary64"] for row in case["samples"])
        print(f"{name}: error upper={error:.6g}; theorem bound={bound:.6g}")
    print("P5/P3: exact REMESH commutation and augmented energy split in 3 cases")
    witness = report["p5_remesh_reduction"]["causal_decay_obstruction"]["runtime"]
    print(
        f"Causal mode: direct REMESH amplitude {witness['observed_mode_amplitude']}; "
        "forward diffusion amplitude 1/2"
    )


if __name__ == "__main__":
    main()
