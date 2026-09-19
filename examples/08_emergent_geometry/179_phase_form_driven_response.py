#!/usr/bin/env python3
"""Example 179: a supplied rotating phase contrast drives a circular form.

Evaluate the analytic particular response on the unit P2 x C3 prism, using
NODAL_PARAMETER_FOUNDATIONS section 16. The primitive phase path and its clock
are imposed. No engine step, glyph, fitted source, or autonomous phase law is
used. Every graph is a newly constructed snapshot of the prescribed curve.

The internal basis P=(1,-1,0), Q=(1,1,-2) has Gram diag(2,6), hence
z=sqrt(2)*u+i*sqrt(6)*v. Prescribe zeta=rho*exp(i*omega*t), then
z=-w*zeta/(pi*(e+i*omega)). The mean is integrated from its actual nonlinear
phasor source; it is never reset. Composite Simpson refinement measures a
quadrature discrepancy, not a rigorous error bound. Binary64 channel and
assembly residuals remain separate from this quadrature comparison.

Run with --output-dir PATH for JSON, CSV and a standard Matplotlib PNG.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tnfr.dynamics.canonical import compute_canonical_nodal_derivative  # noqa: E402
from tnfr.physics.fields import (  # noqa: E402
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length_with_provenance,
)
from tnfr.physics.forcing_realization import capture_non_epi_forcing  # noqa: E402

E, W, VF_WEIGHT = 0.5, 0.25, 0.25
RHO, OMEGA, BETA, MU0 = 0.25, 0.5, math.pi, 0.5
P, Q = np.array([1.0, -1.0, 0.0]), np.array([1.0, 1.0, -2.0])


def contrast(values):
    """Decode metric-normalized complex coordinates into a real triple."""
    values = np.asarray(values)
    return values.real[..., None] * P / math.sqrt(2) + (
        values.imag[..., None] * Q / math.sqrt(6)
    )


def mean_source(times):
    eta = contrast(RHO * np.exp(1j * OMEGA * np.asarray(times)))
    return np.angle(np.sum(np.exp(1j * eta), axis=-1)) / math.pi


def integrated_mean_offset(times, subdivisions):
    """Composite Simpson per interval; retain the offset before adding mu0."""
    grid = np.linspace(times[0], times[-1], (len(times) - 1) * subdivisions + 1)
    values = mean_source(grid)
    increments = []
    for start in range(0, len(grid) - 1, subdivisions):
        block = values[start : start + subdivisions + 1]
        integral = (
            (grid[1] - grid[0])
            * (block[0] + block[-1] + 4 * sum(block[1:-1:2]) + 2 * sum(block[2:-1:2]))
            / 3
        )
        increments.append(W * integral)
    return np.r_[0.0, np.cumsum(increments)]


def snapshot(epi, phase, pressure):
    graph = nx.cartesian_product(nx.path_graph(2), nx.cycle_graph(3))
    graph.graph["DNFR_WEIGHTS"] = {
        "epi": E,
        "phase": W,
        "vf": VF_WEIGHT,
        "topo": 0.0,
    }
    for a, i in graph:
        graph.nodes[a, i].update(
            EPI=float(epi[i]),
            nu_f=1.0,
            theta=float(phase[i]),
            delta_nfr=float(pressure[i]),
        )
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 1.0, "length")
    return graph


def tetrad(graph):
    return {
        "structural_potential": list(
            map(float, compute_structural_potential(graph).values())
        ),
        "phase_gradient": list(map(float, compute_phase_gradient(graph).values())),
        "phase_curvature": list(map(float, compute_phase_curvature(graph).values())),
        "coherence_length": asdict(estimate_coherence_length_with_provenance(graph)),
        "pressure_scope": "declared analytic snapshot pressure; not runtime history",
    }


def run_example():
    times = np.linspace(0.0, 2 * math.pi / OMEGA, 97)
    coarse_offset, mean_offset = (
        integrated_mean_offset(times, count) for count in (16, 32)
    )
    means = MU0 + mean_offset
    rows, endpoint_fields = [], {}
    for index, (time, mean) in enumerate(zip(times, means, strict=True)):
        zeta = RHO * np.exp(1j * OMEGA * time)
        z = -W * zeta / (math.pi * complex(E, OMEGA))
        eta, deviation = contrast(zeta), contrast(z)
        c = float(mean_source(time))
        phase, epi = BETA + eta, mean + deviation
        source = W * (c - eta / math.pi)
        pressure = -E * deviation + source
        derivative = W * c + contrast(1j * OMEGA * z)
        graph = snapshot(epi, phase, pressure)
        observed = capture_non_epi_forcing(graph)
        fresh = np.array([float(p) for p in observed.full_kernel_pressure])
        rate = np.array(
            [
                compute_canonical_nodal_derivative(float(nu), float(p)).derivative
                for nu, p in zip(
                    observed.snapshot.capacity,
                    observed.full_kernel_pressure,
                    strict=True,
                )
            ]
        )
        exact_model = np.array(
            [
                float(observed.epi_weight * g + f)
                for g, f in zip(
                    observed.snapshot.epi_gradient, observed.forcing, strict=True
                )
            ]
        )
        fine_rate = np.tile(derivative, 2)
        work = 6 * float(np.dot(source, derivative))  # D=3I, two triangles
        dissipation = 6 * float(np.dot(derivative, derivative))
        rows.append(
            {
                "time": float(time),
                "z_real": float(z.real),
                "z_imag": float(z.imag),
                "radius": float(abs(z)),
                "mean": float(mean),
                "mean_rate": W * c,
                "mean_offset": float(mean_offset[index]),
                "mean_quadrature_difference": float(
                    mean_offset[index] - coarse_offset[index]
                ),
                "epi": np.tile(epi, 2).tolist(),
                "primitive_phase": np.tile(phase, 2).tolist(),
                "analytic_pressure": np.tile(pressure, 2).tolist(),
                "analytic_rate": fine_rate.tolist(),
                "fresh_kernel_pressure": fresh.tolist(),
                "fresh_nodal_rate": rate.tolist(),
                "represented_model_pressure": exact_model.tolist(),
                "kernel_assembly_defect_exact": [
                    str(q) for q in observed.kernel_pressure_defect
                ],
                "stored_minus_fresh_exact": [
                    str(q) for q in observed.stored_pressure_residual
                ],
                "max_kernel_assembly_defect": max(
                    float(abs(q)) for q in observed.kernel_pressure_defect
                ),
                "max_stored_pressure_residual": max(
                    float(abs(q)) for q in observed.stored_pressure_residual
                ),
                "max_nodal_rate_residual": float(max(abs(rate - fine_rate))),
                "max_model_vs_analytic": float(max(abs(exact_model - fine_rate))),
                "source_work": work,
                "metric_speed_squared": dissipation,
                "work_balance_residual": work - dissipation,
                "internal_source_work": float(
                    6 * np.real(np.conj(z) * (-W * zeta / math.pi))
                ),
                "internal_dissipation": float(6 * E * abs(z) ** 2),
            }
        )
        if index in (0, len(times) - 1):
            endpoint_fields[str(index)] = tetrad(graph)
    summary = {
        "radius": rows[0]["radius"],
        "mean_peak_to_peak": float(np.ptp(mean_offset)),
        "mean_return_residual": float(means[-1] - means[0]),
        "raw_accumulated_endpoint_offsets": [
            float(coarse_offset[-1]),
            float(mean_offset[-1]),
        ],
        "quadrature_refinement_max_difference": float(
            max(abs(mean_offset - coarse_offset))
        ),
        "max_nodal_rate_residual": max(row["max_nodal_rate_residual"] for row in rows),
        "max_kernel_assembly_defect": max(
            row["max_kernel_assembly_defect"] for row in rows
        ),
        "max_stored_pressure_residual": max(
            row["max_stored_pressure_residual"] for row in rows
        ),
        "max_work_balance_residual": max(
            abs(row["work_balance_residual"]) for row in rows
        ),
        "sampled_epi_range": [
            min(min(row["epi"]) for row in rows),
            max(max(row["epi"]) for row in rows),
        ],
        "phase_gap_analytic_upper_bound": math.sqrt(2) * RHO,
    }
    # Numerical demonstration checks, not exact-real or runtime certificates.
    if summary["max_nodal_rate_residual"] > 1e-12:
        raise ArithmeticError(
            "shared nodal rate disagrees with supplied analytic response"
        )
    if summary["max_work_balance_residual"] > 1e-12:
        raise ArithmeticError("analytic source-work balance failed")
    return {
        "version": 1,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
        },
        "parameters": {
            "e": E,
            "w_phase": W,
            "w_vf": VF_WEIGHT,
            "rho": RHO,
            "omega": OMEGA,
            "beta": BETA,
            "mu0": MU0,
            "alpha0": 0.0,
        },
        "scope": [
            "supplied primitive phase path and external clock",
            "analytic particular EPI response; independent fresh snapshot pressure readouts",
            "uniform unit capacity makes the nonzero vf weight's channel vanish",
            "fixed unit conductances and explicit edge lengths; no runtime or glyph evolution",
            "mean uses accumulated Simpson quadrature, never a per-sample reset",
            "refinement discrepancy is not a rigorous quadrature error bound",
            "tetrad is diagnostic, not state closure or evidence of autonomous maintenance",
        ],
        "quadrature": {"output_intervals": 96, "subdivisions": [16, 32]},
        "summary": summary,
        "rows": rows,
        "endpoint_tetrad": endpoint_fields,
    }


def save_outputs(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / "phase_form_driven_response"
    stem.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    columns = (
        "time",
        "z_real",
        "z_imag",
        "radius",
        "mean",
        "mean_offset",
        "mean_rate",
        "mean_quadrature_difference",
        "max_nodal_rate_residual",
        "source_work",
        "metric_speed_squared",
        "internal_source_work",
        "internal_dissipation",
    )
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["rows"])
    rows = report["rows"]
    time = np.array([row["time"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].plot([r["z_real"] for r in rows], [r["z_imag"] for r in rows])
    axes[0].set(
        xlabel="Re z", ylabel="Im z", title="Internal circular response", aspect="equal"
    )
    axes[1].plot(time, [r["mean_offset"] * 1e6 for r in rows])
    axes[1].set(
        xlabel="Supplied time", ylabel="(mean − mean₀) × 10⁶", title="Small moving mean"
    )
    axes[2].plot(time, [r["internal_source_work"] for r in rows], label="Source work")
    axes[2].plot(
        time, [r["internal_dissipation"] for r in rows], "--", label="Radial loss"
    )
    axes[2].set(
        xlabel="Supplied time",
        ylabel="H-weighted variance / time",
        title="Constant amplitude balance",
    )
    axes[2].legend()
    fig.suptitle("Imposed rotating primitive phase; conditional analytic response")
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = run_example()
    if args.output_dir is not None:
        save_outputs(report, args.output_dir)
    print(
        json.dumps({"scope": report["scope"], "summary": report["summary"]}, indent=2)
    )


if __name__ == "__main__":
    main()
