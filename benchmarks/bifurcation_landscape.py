"""Bifurcation landscape benchmark (Phase 5 REAL metrics).

Sweeps destabilizer (OZ) intensity and mutation (ZHIR) thresholds across
selected topologies; computes physics-grounded bifurcation metrics:
ΔΦ_s, Δ|∇φ|_max, ΔK_φ_max, ξ_C amplification, ΔVar(ΔNFR), bifurcation score.

Physics alignment:
- Uses ONLY canonical operators to mutate structure (Emission, Dissonance, Coherence, SelfOrganization, Mutation).
- Field computations are read-only telemetry (Φ_s, |∇φ|, K_φ, ξ_C) preserving Invariant #1.
- Handlers_present reflects U4a compliance (IL/THOL after OZ/ZHIR where applicable).

CLI Flags:
  --nodes <int>
  --seeds <int>
  --topologies ring,ws,scale_free,grid
  --oz-intensity-grid 0.5,1.0,1.5
  --mutation-thresholds 0.25,0.5
  --vf-grid 0.5,0.75
  --bifurcation-score-threshold 0.5
  --phase-gradient-spike 0.12
  --phase-curvature-spike 0.15
  --coherence-length-amplification 1.5
  --dnfr-variance-increase 0.2
  --structural-potential-shift 0.3
  --fragmentation-coherence-threshold 0.3
  --dry-run
  --quiet

Output JSONL keys:
  topology, intensity_oz, mutation_threshold, vf_scale, seed,
  delta_phi_s, delta_phase_gradient_max, delta_phase_curvature_max,
  coherence_length_ratio, delta_dnfr_variance, bifurcation_score_max,
  handlers_present, classification (none|incipient|bifurcation|fragmentation)

English-only per language policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

# Handle both module and script execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent))
    from bifurcation_metrics import (  # type: ignore
        build_topology,
        initialize_graph_state,
        capture_fields,
        apply_bifurcation_sequence,
        compute_bifurcation_metrics,
    )
else:
    from .bifurcation_metrics import (
        build_topology,
        initialize_graph_state,
        capture_fields,
        apply_bifurcation_sequence,
        compute_bifurcation_metrics,
    )

PROJECT_ROOT = Path(__file__).parent.parent


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def bifurcation_landscape(
    topologies: Iterable[str],
    n_nodes: int,
    oz_grid: List[float],
    mutation_thresholds: List[float],
    vf_grid: List[float],
    n_seeds: int,
    bifurcation_score_threshold: float,
    phase_gradient_spike: float,
    phase_curvature_spike: float,
    coherence_length_amplification: float,
    dnfr_variance_increase: float,
    structural_potential_shift: float,
    fragmentation_coherence_threshold: float,
    dry_run: bool = False,
    quiet: bool = False,
) -> int:
    if dry_run:
        if not quiet:
            print("[dry-run] Parameters validated:")
            print(
                json.dumps(
                    {
                        "topologies": list(topologies),
                        "nodes": n_nodes,
                        "oz_grid": oz_grid,
                        "mutation_thresholds": mutation_thresholds,
                        "vf_grid": vf_grid,
                        "bifurcation_score_threshold": bifurcation_score_threshold,
                        "phase_gradient_spike": phase_gradient_spike,
                        "phase_curvature_spike": phase_curvature_spike,
                        "coherence_length_amplification": coherence_length_amplification,
                        "dnfr_variance_increase": dnfr_variance_increase,
                        "structural_potential_shift": structural_potential_shift,
                        "fragmentation_coherence_threshold": fragmentation_coherence_threshold,
                        "seeds": n_seeds,
                    },
                    indent=2,
                )
            )
        return 0

    record_count = 0
    for topo in topologies:
        for intensity in oz_grid:
            for mut_thr in mutation_thresholds:
                for vf_scale in vf_grid:
                    for seed_idx in range(n_seeds):
                        G = build_topology(topo, n_nodes, seed=seed_idx)
                        initialize_graph_state(G, vf_scale=vf_scale, seed=seed_idx)
                        pre = capture_fields(G)
                        handlers_present = apply_bifurcation_sequence(
                            G,
                            intensity_oz=intensity,
                            mutation_threshold=mut_thr,
                            vf_scale=vf_scale,
                            seed=seed_idx,
                        )
                        post = capture_fields(G)
                        metrics = compute_bifurcation_metrics(
                            G,
                            pre,
                            post,
                            bifurcation_score_threshold=bifurcation_score_threshold,
                            phase_gradient_spike=phase_gradient_spike,
                            phase_curvature_spike=phase_curvature_spike,
                            coherence_length_amplification=coherence_length_amplification,
                            dnfr_variance_increase=dnfr_variance_increase,
                            structural_potential_shift=structural_potential_shift,
                            fragmentation_coherence_threshold=fragmentation_coherence_threshold,
                            handlers_present=handlers_present,
                        )
                        record = {
                            "topology": topo,
                            "intensity_oz": intensity,
                            "mutation_threshold": mut_thr,
                            "vf_scale": vf_scale,
                            "seed": seed_idx,
                            **metrics,
                        }
                        if not quiet:
                            print(json.dumps(record))
                        record_count += 1
    if not quiet:
        print(f"Completed bifurcation sweep (records={record_count}).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 5 bifurcation landscape benchmark (real metrics)",
    )
    p.add_argument("--nodes", type=int, default=50,
                   help="Nodes per topology")
    p.add_argument("--seeds", type=int, default=3, help="Seeds per grid point")
    p.add_argument("--topologies", type=str,
                   default="ring,ws,scale_free",
                   help="Topologies list")
    p.add_argument("--oz-intensity-grid", type=str,
                   default="0.5,1.0,1.5",
                   help="OZ intensity values")
    p.add_argument("--mutation-thresholds", type=str,
                   default="0.25,0.5",
                   help="Mutation thresholds")
    p.add_argument("--vf-grid", type=str,
                   default="0.5,0.75",
                   help="VF scaling factors")
    p.add_argument("--bifurcation-score-threshold", type=float,
                   default=0.5,
                   help="Bifurcation score threshold")
    p.add_argument("--phase-gradient-spike", type=float,
                   default=0.12,
                   help="Phase gradient spike Δ|∇φ|")
    p.add_argument("--phase-curvature-spike", type=float,
                   default=0.15,
                   help="Phase curvature spike ΔK_φ")
    p.add_argument("--coherence-length-amplification", type=float,
                   default=1.5,
                   help="ξ_C amplification ratio")
    p.add_argument("--dnfr-variance-increase", type=float,
                   default=0.2,
                   help="ΔVar(ΔNFR) threshold")
    p.add_argument("--structural-potential-shift", type=float,
                   default=0.3,
                   help="|ΔΦ_s| mean abs shift")
    p.add_argument("--fragmentation-coherence-threshold", type=float,
                   default=0.3,
                   help="Fragmentation coherence threshold")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate parameters only")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress stdout")
    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    oz_grid = _parse_float_list(args.oz_intensity_grid)
    mutation_thresholds = _parse_float_list(args.mutation_thresholds)
    vf_grid = _parse_float_list(args.vf_grid)
    return bifurcation_landscape(
        topologies=topologies,
        n_nodes=args.nodes,
        oz_grid=oz_grid,
        mutation_thresholds=mutation_thresholds,
        vf_grid=vf_grid,
        n_seeds=args.seeds,
        bifurcation_score_threshold=args.bifurcation_score_threshold,
        phase_gradient_spike=args.phase_gradient_spike,
        phase_curvature_spike=args.phase_curvature_spike,
        coherence_length_amplification=args.coherence_length_amplification,
        dnfr_variance_increase=args.dnfr_variance_increase,
        structural_potential_shift=args.structural_potential_shift,
        fragmentation_coherence_threshold=(
            args.fragmentation_coherence_threshold
        ),
        dry_run=args.dry_run,
        quiet=args.quiet,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
