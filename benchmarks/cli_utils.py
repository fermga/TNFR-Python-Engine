"""CLI utilities for TNFR benchmarks with scaling capabilities.

Provides common command-line argument parsing for:
- Network sizes (--nodes)
- Topology types (--topologies)
- Random seeds (--seeds, --seed-count)
- Parameter grids (--param-grid)
- Output control (--output-dir, --format)
- Precision/telemetry modes (--precision, --telemetry)

Usage:
    parser = create_benchmark_parser(
        description="My benchmark",
        default_nodes=50,
        default_seeds=10
    )
    args = parser.parse_args()
    
Physics Invariance:
- CLI flags control ONLY scale/sampling, NOT grammar or operators
- Precision/telemetry modes from Phase 1-3 are read-only
- All U1-U6 invariants preserved
"""

import argparse
from pathlib import Path
from typing import List, Optional


def create_benchmark_parser(
    description: str,
    default_nodes: int = 50,
    default_seeds: int = 10,
    default_topologies: Optional[List[str]] = None,
    add_precision_flags: bool = True,
    add_param_grid: bool = False,
) -> argparse.ArgumentParser:
    """Create standard argument parser for TNFR benchmarks.
    
    Parameters
    ----------
    description : str
        Benchmark description for help text
    default_nodes : int
        Default number of nodes (can be overridden with --nodes)
    default_seeds : int
        Default number of random seeds (can be overridden with --seed-count)
    default_topologies : list of str, optional
        Default topology types. If None, uses ["ws", "scale_free"]
    add_precision_flags : bool
        Whether to add --precision and --telemetry flags
    add_param_grid : bool
        Whether to add --param-grid for fine parameter sweeps
    
    Returns
    -------
    argparse.ArgumentParser
        Configured parser ready for parse_args()
    """
    if default_topologies is None:
        default_topologies = ["ws", "scale_free"]
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Network scale
    parser.add_argument(
        "--nodes",
        type=int,
        default=default_nodes,
        help="Number of nodes in network (scales experiment size)",
    )
    
    parser.add_argument(
        "--nodes-list",
        type=int,
        nargs="+",
        help="Run experiment for multiple network sizes (overrides --nodes)",
    )
    
    # Topology types
    parser.add_argument(
        "--topologies",
        type=str,
        nargs="+",
        default=default_topologies,
        choices=["ring", "ws", "scale_free", "grid", "er"],
        help="Topology types to test",
    )
    
    # Random seeds
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility",
    )
    
    parser.add_argument(
        "--seed-count",
        type=int,
        default=default_seeds,
        help="Number of random seeds (runs per configuration)",
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Explicit list of seeds (overrides --seed-count)",
    )
    
    # Output control
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files",
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "csv", "both"],
        default="json",
        help="Output file format",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (only show final results)",
    )
    
    # Precision and telemetry (Phase 1-3 integration)
    if add_precision_flags:
        parser.add_argument(
            "--precision",
            type=str,
            choices=["standard", "high", "research"],
            default="standard",
            help="Precision mode for numerical computations (Phase 1-2)",
        )
        
        parser.add_argument(
            "--telemetry",
            type=str,
            choices=["low", "medium", "high"],
            default="low",
            help="Telemetry density for snapshot collection (Phase 3)",
        )
    
    # Parameter grid support
    if add_param_grid:
        parser.add_argument(
            "--param-grid-resolution",
            type=str,
            choices=["coarse", "medium", "fine"],
            default="medium",
            help="Parameter grid resolution around critical points",
        )
        
        parser.add_argument(
            "--param-range",
            type=float,
            nargs=2,
            metavar=("MIN", "MAX"),
            help="Custom parameter range (min max)",
        )
    
    return parser


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    """Resolve seed list from CLI arguments.
    
    Priority:
    1. Explicit --seeds list
    2. Generate from --seed-count starting at --seed
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from create_benchmark_parser()
    
    Returns
    -------
    list of int
        List of random seeds to use
    """
    if hasattr(args, "seeds") and args.seeds:
        return args.seeds
    
    seed_count = args.seed_count if hasattr(args, "seed_count") else 10
    base_seed = args.seed if hasattr(args, "seed") else 42
    
    return [base_seed + i for i in range(seed_count)]


def resolve_node_sizes(args: argparse.Namespace) -> List[int]:
    """Resolve list of network sizes from CLI arguments.
    
    Priority:
    1. Explicit --nodes-list
    2. Single --nodes value
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from create_benchmark_parser()
    
    Returns
    -------
    list of int
        List of network sizes to test
    """
    if hasattr(args, "nodes_list") and args.nodes_list:
        return args.nodes_list
    
    nodes = args.nodes if hasattr(args, "nodes") else 50
    return [nodes]


def setup_output_dir(args: argparse.Namespace) -> Path:
    """Create output directory if needed.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with output_dir attribute
    
    Returns
    -------
    Path
        Path to output directory (created if doesn't exist)
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def apply_precision_config(args: argparse.Namespace) -> None:
    """Apply precision and telemetry modes from CLI args.
    
    Uses Phase 1-3 configuration system.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with precision/telemetry attributes
    """
    if not hasattr(args, "precision"):
        return
    
    try:
        from tnfr.config import (
            set_precision_mode,
            set_telemetry_density,
        )
        
        if args.precision:
            set_precision_mode(args.precision)
        
        if hasattr(args, "telemetry") and args.telemetry:
            set_telemetry_density(args.telemetry)
            
    except ImportError:
        # Graceful degradation if config not available
        pass


def get_param_grid_points(
    resolution: str,
    critical_point: float,
    param_range: Optional[tuple] = None,
) -> List[float]:
    """Generate parameter grid around critical point.
    
    Parameters
    ----------
    resolution : str
        "coarse" (10 points) | "medium" (25 points) | "fine" (50 points)
    critical_point : float
        Critical parameter value (e.g., I_c = 2.015)
    param_range : tuple of (min, max), optional
        Custom range. If None, uses critical_point ± 20%
    
    Returns
    -------
    list of float
        Parameter values to sample
    """
    import numpy as np
    
    if param_range:
        min_val, max_val = param_range
    else:
        # Default: ±20% around critical point
        min_val = critical_point * 0.8
        max_val = critical_point * 1.2
    
    # Resolution determines number of points
    n_points = {
        "coarse": 10,
        "medium": 25,
        "fine": 50,
    }.get(resolution, 25)
    
    # Denser sampling near critical point
    # Use log spacing on both sides
    below_points = np.linspace(min_val, critical_point, n_points // 2)
    above_points = np.linspace(
        critical_point, max_val, n_points // 2 + 1
    )[1:]  # Avoid duplicate at critical_point
    
    return np.concatenate([below_points, above_points]).tolist()
