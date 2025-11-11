"""Argument parser helpers shared across TNFR CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from ..config.presets import PREFERRED_PRESET_NAMES
from ..gamma import GAMMA_REGISTRY
from ..telemetry.verbosity import TELEMETRY_VERBOSITY_LEVELS
from ..types import ArgSpec
from .utils import spec

_PRESET_HELP = "Available presets: {}.".format(
    ", ".join(PREFERRED_PRESET_NAMES),
)

TELEMETRY_VERBOSITY_CHOICES = TELEMETRY_VERBOSITY_LEVELS

GRAMMAR_ARG_SPECS: tuple[ArgSpec, ...] = (
    spec("--grammar.enabled", action=argparse.BooleanOptionalAction),
    spec("--grammar.zhir_requires_oz_window", type=int),
    spec("--grammar.zhir_dnfr_min", type=float),
    spec("--grammar.thol_min_len", type=int),
    spec("--grammar.thol_max_len", type=int),
    spec("--grammar.thol_close_dnfr", type=float),
    spec("--grammar.si_high", type=float),
    spec("--glyph.hysteresis_window", type=int),
)

# History export/save specifications
HISTORY_ARG_SPECS: tuple[ArgSpec, ...] = (
    spec("--save-history", type=str),
    spec("--export-history-base", type=str),
    spec("--export-format", choices=["csv", "json"], default="json"),
)

# Arguments shared by CLI subcommands
COMMON_ARG_SPECS: tuple[ArgSpec, ...] = (
    spec("--nodes", type=int, default=24),
    spec("--topology", choices=["ring", "complete", "erdos"], default="ring"),
    spec("--seed", type=int, default=1),
    spec(
        "--p",
        type=float,
        help="Edge probability when topology=erdos",
    ),
    spec("--observer", action="store_true", help="Attach standard observer"),
    spec(
        "--trace-verbosity",
        choices=TELEMETRY_VERBOSITY_CHOICES,
        help="Select the trace capture preset",
    ),
    spec(
        "--metrics-verbosity",
        choices=TELEMETRY_VERBOSITY_CHOICES,
        help="Select the metrics capture preset",
    ),
    spec("--config", type=str),
    spec("--dt", type=float),
    spec("--integrator", choices=["euler", "rk4"]),
    spec("--remesh-mode", choices=["knn", "mst", "community"]),
    spec("--gamma-type", choices=list(GAMMA_REGISTRY.keys()), default="none"),
    spec("--gamma-beta", type=float, default=0.0),
    spec("--gamma-R0", type=float, default=0.0),
    spec("--um-candidate-count", type=int),
    spec("--stop-early-window", type=int),
    spec("--stop-early-fraction", type=float),
)


def add_arg_specs(parser: argparse._ActionsContainer, specs: Iterable[ArgSpec]) -> None:
    """Register arguments from ``specs`` on ``parser``."""
    for opt, kwargs in specs:
        parser.add_argument(opt, **kwargs)


def _args_to_dict(args: argparse.Namespace, prefix: str) -> dict[str, Any]:
    """Extract arguments matching a prefix."""
    return {
        k.removeprefix(prefix): v
        for k, v in vars(args).items()
        if k.startswith(prefix) and v is not None
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across subcommands."""
    add_arg_specs(parser, COMMON_ARG_SPECS)


def add_grammar_args(parser: argparse.ArgumentParser) -> None:
    """Add grammar and structural operator hysteresis options."""
    group = parser.add_argument_group("Grammar")
    add_arg_specs(group, GRAMMAR_ARG_SPECS)


def add_grammar_selector_args(parser: argparse.ArgumentParser) -> None:
    """Add grammar options and structural operator selector."""
    add_grammar_args(parser)
    parser.add_argument("--selector", choices=["basic", "param"], default="basic")


def add_history_export_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments to save or export history."""
    add_arg_specs(parser, HISTORY_ARG_SPECS)


def add_canon_toggle(parser: argparse.ArgumentParser) -> None:
    """Add option to disable canonical grammar."""
    parser.add_argument(
        "--no-canon",
        dest="grammar_canon",
        action="store_false",
        default=True,
        help="Disable canonical grammar",
    )


def _add_run_parser(sub: argparse._SubParsersAction) -> None:
    """Configure the ``run`` subcommand."""

    from .execution import DEFAULT_SUMMARY_SERIES_LIMIT, cmd_run

    p_run = sub.add_parser(
        "run",
        help=("Run a free scenario or preset and optionally export history"),
    )
    add_common_args(p_run)
    p_run.add_argument("--steps", type=int, default=100)
    p_run.add_argument(
        "--use-Si",
        dest="use_Si",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Recompute the Sense Index during the run (use --no-use-Si to disable)",
    )
    p_run.add_argument(
        "--apply-glyphs",
        dest="apply_glyphs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply structural operators at every step (use --no-apply-glyphs to disable)",
    )
    p_run.add_argument(
        "--dnfr-n-jobs",
        dest="dnfr_n_jobs",
        type=int,
        help="Override ΔNFR parallel jobs forwarded to the runtime",
    )
    add_canon_toggle(p_run)
    add_grammar_selector_args(p_run)
    add_history_export_args(p_run)
    p_run.add_argument("--preset", type=str, default=None, help=_PRESET_HELP)
    p_run.add_argument("--sequence-file", type=str, default=None)
    p_run.add_argument("--summary", action="store_true")
    p_run.add_argument(
        "--summary-limit",
        type=int,
        default=DEFAULT_SUMMARY_SERIES_LIMIT,
        help=("Maximum number of samples per series in the summary (<=0 to" " disable trimming)"),
    )

    math_group = p_run.add_argument_group("Mathematical dynamics")
    math_group.add_argument(
        "--math-engine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the spectral mathematical dynamics engine to project nodes"
            " onto Hilbert space vectors and validate norm, coherence and"
            " structural frequency invariants"
        ),
    )
    math_group.add_argument(
        "--math-dimension",
        type=int,
        help="Hilbert space dimension to use when the math engine is enabled",
    )
    math_group.add_argument(
        "--math-coherence-spectrum",
        type=float,
        nargs="+",
        metavar="λ",
        help=(
            "Eigenvalues for the coherence operator (defaults to a flat" " spectrum when omitted)"
        ),
    )
    math_group.add_argument(
        "--math-coherence-c-min",
        type=float,
        help="Explicit coherence floor C_min for the operator",
    )
    math_group.add_argument(
        "--math-coherence-threshold",
        type=float,
        help="Coherence expectation threshold enforced during validation",
    )
    math_group.add_argument(
        "--math-frequency-diagonal",
        type=float,
        nargs="+",
        metavar="ν",
        help=(
            "Diagonal entries for the structural frequency operator"
            " (defaults to the identity spectrum)"
        ),
    )
    math_group.add_argument(
        "--math-generator-diagonal",
        type=float,
        nargs="+",
        metavar="ω",
        help=(
            "Diagonal ΔNFR generator used by the mathematical dynamics"
            " engine (defaults to the null generator)"
        ),
    )

    p_run.set_defaults(func=cmd_run)


def _add_sequence_parser(sub: argparse._SubParsersAction) -> None:
    """Configure the ``sequence`` subcommand."""
    from .execution import cmd_sequence

    p_seq = sub.add_parser(
        "sequence",
        help="Execute a sequence (preset or YAML/JSON)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "JSON sequence example:\n"
            "[\n"
            '  "A",\n'
            '  {"WAIT": 1},\n'
            '  {"THOL": {"body": ["A", {"WAIT": 2}], "repeat": 2}}\n'
            "]"
        ),
    )
    add_common_args(p_seq)
    p_seq.add_argument("--preset", type=str, default=None, help=_PRESET_HELP)
    p_seq.add_argument("--sequence-file", type=str, default=None)
    add_history_export_args(p_seq)
    add_grammar_args(p_seq)
    p_seq.set_defaults(func=cmd_sequence)


def _add_metrics_parser(sub: argparse._SubParsersAction) -> None:
    """Configure the ``metrics`` subcommand."""
    from .execution import cmd_metrics

    p_met = sub.add_parser("metrics", help="Run briefly and export key metrics")
    add_common_args(p_met)
    p_met.add_argument("--steps", type=int, default=None)
    add_canon_toggle(p_met)
    add_grammar_selector_args(p_met)
    p_met.add_argument("--save", type=str, default=None)
    p_met.add_argument(
        "--summary-limit",
        type=int,
        default=None,
        help=("Maximum number of samples per series in the summary (<=0 to" " disable trimming)"),
    )
    p_met.set_defaults(func=cmd_metrics)


def _add_profile_parser(sub: argparse._SubParsersAction) -> None:
    """Configure the ``profile-si`` subcommand."""

    from .execution import cmd_profile_si

    p_prof = sub.add_parser(
        "profile-si",
        help="Profile compute_Si with and without NumPy",
    )
    p_prof.add_argument("--nodes", type=int, default=240)
    p_prof.add_argument("--chord-step", type=int, default=7)
    p_prof.add_argument("--loops", type=int, default=5)
    p_prof.add_argument("--output-dir", type=Path, default=Path("profiles"))
    p_prof.add_argument("--format", choices=("pstats", "json"), default="pstats")
    p_prof.add_argument("--sort", choices=("cumtime", "tottime"), default="cumtime")
    p_prof.set_defaults(func=cmd_profile_si)


def _add_profile_pipeline_parser(sub: argparse._SubParsersAction) -> None:
    """Configure the ``profile-pipeline`` subcommand."""

    from .execution import cmd_profile_pipeline

    p_profile = sub.add_parser(
        "profile-pipeline",
        help="Profile the Sense Index + ΔNFR pipeline",
    )
    p_profile.add_argument("--nodes", type=int, default=240, help="Number of nodes")
    p_profile.add_argument(
        "--edge-probability",
        type=float,
        default=0.32,
        help="Probability passed to the Erdos-Renyi generator",
    )
    p_profile.add_argument(
        "--loops",
        type=int,
        default=5,
        help="How many times to execute the pipeline inside the profiler",
    )
    p_profile.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when generating the graph",
    )
    p_profile.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profiles"),
        help="Directory where profiling artefacts will be written",
    )
    p_profile.add_argument(
        "--sort",
        choices=("cumtime", "tottime"),
        default="cumtime",
        help="Sort order applied to profiling rows",
    )
    p_profile.add_argument(
        "--si-chunk-sizes",
        nargs="+",
        metavar="SIZE",
        help="Chunk sizes forwarded to G.graph['SI_CHUNK_SIZE']; use 'auto' for heuristics",
    )
    p_profile.add_argument(
        "--dnfr-chunk-sizes",
        nargs="+",
        metavar="SIZE",
        help="Chunk sizes forwarded to G.graph['DNFR_CHUNK_SIZE']; use 'auto' for heuristics",
    )
    p_profile.add_argument(
        "--si-workers",
        nargs="+",
        metavar="COUNT",
        help="Worker counts forwarded to G.graph['SI_N_JOBS']; use 'auto' for serial runs",
    )
    p_profile.add_argument(
        "--dnfr-workers",
        nargs="+",
        metavar="COUNT",
        help="Worker counts forwarded to G.graph['DNFR_N_JOBS']; use 'auto' for defaults",
    )
    p_profile.set_defaults(func=cmd_profile_pipeline)


def _add_math_run_parser(sub: argparse._SubParsersAction) -> None:
    """Configure the ``math.run`` subcommand."""
    from .execution import cmd_math_run

    p_math = sub.add_parser(
        "math.run",
        help="Run simulation with mathematical dynamics engine validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run with default math engine settings\n"
            "  tnfr math.run --nodes 24 --steps 100\n\n"
            "  # Run with custom Hilbert dimension\n"
            "  tnfr math.run --math-dimension 32 --steps 50\n\n"
            "  # Run with custom coherence spectrum\n"
            "  tnfr math.run --math-coherence-spectrum 1.0 0.8 0.6 --steps 100\n"
        ),
    )
    add_common_args(p_math)
    p_math.add_argument("--steps", type=int, default=100)
    p_math.add_argument(
        "--preset",
        type=str,
        default=None,
        help=_PRESET_HELP,
    )
    p_math.add_argument("--sequence-file", type=str, default=None)
    add_canon_toggle(p_math)
    add_grammar_selector_args(p_math)
    add_history_export_args(p_math)

    # Math engine is always enabled for math.run
    math_group = p_math.add_argument_group("Mathematical dynamics (always enabled)")
    math_group.add_argument(
        "--math-dimension",
        type=int,
        help="Hilbert space dimension",
    )
    math_group.add_argument(
        "--math-coherence-spectrum",
        type=float,
        nargs="+",
        metavar="λ",
        help="Eigenvalues for the coherence operator",
    )
    math_group.add_argument(
        "--math-coherence-c-min",
        type=float,
        help="Explicit coherence floor C_min",
    )
    math_group.add_argument(
        "--math-coherence-threshold",
        type=float,
        help="Coherence threshold for validation",
    )
    math_group.add_argument(
        "--math-frequency-diagonal",
        type=float,
        nargs="+",
        metavar="ν",
        help="Diagonal entries for the frequency operator",
    )
    math_group.add_argument(
        "--math-generator-diagonal",
        type=float,
        nargs="+",
        metavar="ω",
        help="ΔNFR generator diagonal",
    )

    p_math.set_defaults(func=cmd_math_run)


def _add_epi_validate_parser(sub: argparse._SubParsersAction) -> None:
    """Configure the ``epi.validate`` subcommand."""
    from .execution import cmd_epi_validate

    p_epi = sub.add_parser(
        "epi.validate",
        help="Validate EPI structural integrity and coherence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Validate a preset\n"
            "  tnfr epi.validate --preset resonant_bootstrap\n\n"
            "  # Validate with custom topology\n"
            "  tnfr epi.validate --nodes 48 --topology complete\n\n"
            "  # Validate from sequence file\n"
            "  tnfr epi.validate --sequence-file presets/resonant_bootstrap.yaml\n"
        ),
    )
    add_common_args(p_epi)
    p_epi.add_argument("--steps", type=int, default=50)
    p_epi.add_argument(
        "--preset",
        type=str,
        default=None,
        help=_PRESET_HELP,
    )
    p_epi.add_argument("--sequence-file", type=str, default=None)
    add_canon_toggle(p_epi)
    add_grammar_selector_args(p_epi)

    validation_group = p_epi.add_argument_group("Validation options")
    validation_group.add_argument(
        "--check-coherence",
        action="store_true",
        default=True,
        help="Validate coherence preservation (enabled by default)",
    )
    validation_group.add_argument(
        "--check-frequency",
        action="store_true",
        default=True,
        help="Validate structural frequency positivity (enabled by default)",
    )
    validation_group.add_argument(
        "--check-phase",
        action="store_true",
        default=True,
        help="Validate phase synchrony in couplings (enabled by default)",
    )
    validation_group.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Numerical tolerance for validation checks",
    )

    p_epi.set_defaults(func=cmd_epi_validate)
