"""CLI execution helpers for running canonical TNFR programs."""

from __future__ import annotations

import argparse
import math
from collections import deque
from collections.abc import Iterable, Mapping, Sized
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Sequence

import networkx as nx
import numpy as np

from ..alias import get_attr
from ..config import apply_config
from ..config.presets import (
    PREFERRED_PRESET_NAMES,
    get_preset,
)
from ..constants import METRIC_DEFAULTS, VF_PRIMARY, get_aliases, get_param
from ..dynamics import default_glyph_selector, parametric_glyph_selector, run
from ..execution import CANONICAL_PRESET_NAME, play
from ..flatten import parse_program_tokens
from ..glyph_history import ensure_history
from ..mathematics import (
    BasicStateProjector,
    CoherenceOperator,
    FrequencyOperator,
    HilbertSpace,
    MathematicalDynamicsEngine,
    make_coherence_operator,
    make_frequency_operator,
)
from ..metrics import (
    build_metrics_summary,
    export_metrics,
    glyph_top,
    register_metrics_callbacks,
)
from ..metrics.core import _metrics_step
from ..ontosim import prepare_network
from ..sense import register_sigma_callback
from ..trace import register_trace
from ..types import ProgramTokens
from ..utils import (
    StructuredFileError,
    clamp01,
    get_logger,
    json_dumps,
    read_structured_file,
    safe_write,
)
from ..validation import NFRValidator, validate_canon
from .arguments import _args_to_dict
from .utils import _parse_cli_variants

# Constants
TWO_PI = 2 * math.pi

logger = get_logger(__name__)

_VF_ALIASES = get_aliases("VF")
VF_ALIAS_KEYS: tuple[str, ...] = (VF_PRIMARY,) + tuple(
    alias for alias in _VF_ALIASES if alias != VF_PRIMARY
)

_EPI_ALIASES = get_aliases("EPI")
EPI_PRIMARY = _EPI_ALIASES[0]
EPI_ALIAS_KEYS: tuple[str, ...] = (EPI_PRIMARY,) + tuple(
    alias for alias in _EPI_ALIASES if alias != EPI_PRIMARY
)

# CLI summaries should remain concise by default while allowing callers to
# inspect the full glyphogram series when needed.
DEFAULT_SUMMARY_SERIES_LIMIT = 10

_PREFERRED_PRESETS_DISPLAY = ", ".join(PREFERRED_PRESET_NAMES)


def _as_iterable_view(view: Any) -> Iterable[Any]:
    """Return ``view`` as an iterable, resolving callable cached views."""

    if hasattr(view, "__iter__"):
        return view  # type: ignore[return-value]
    if callable(view):
        resolved = view()
        if not hasattr(resolved, "__iter__"):
            raise TypeError("Graph view did not return an iterable")
        return resolved
    return ()


def _iter_graph_nodes(graph: Any) -> Iterable[Any]:
    """Yield nodes from ``graph`` normalising NetworkX-style accessors."""

    return _as_iterable_view(getattr(graph, "nodes", ()))


def _iter_graph_edges(graph: Any) -> Iterable[Any]:
    """Yield edges from ``graph`` normalising NetworkX-style accessors."""

    return _as_iterable_view(getattr(graph, "edges", ()))


def _count_graph_nodes(graph: Any) -> int:
    """Return node count honouring :class:`tnfr.types.GraphLike` semantics."""

    if hasattr(graph, "number_of_nodes"):
        return int(graph.number_of_nodes())
    nodes_view = _iter_graph_nodes(graph)
    if isinstance(nodes_view, Sized):
        return len(nodes_view)  # type: ignore[arg-type]
    return len(tuple(nodes_view))


def _save_json(path: str, data: Any) -> None:
    payload = json_dumps(data, ensure_ascii=False, indent=2, default=list)
    safe_write(path, lambda f: f.write(payload))


def _attach_callbacks(G: "nx.Graph") -> None:
    register_sigma_callback(G)
    register_metrics_callbacks(G)
    register_trace(G)
    history = ensure_history(G)
    maxlen = int(get_param(G, "PROGRAM_TRACE_MAXLEN"))
    history.setdefault("program_trace", deque(maxlen=maxlen))
    history.setdefault("trace_meta", [])
    _metrics_step(G, ctx=None)


def _persist_history(G: "nx.Graph", args: argparse.Namespace) -> None:
    if getattr(args, "save_history", None) or getattr(args, "export_history_base", None):
        history = ensure_history(G)
        if getattr(args, "save_history", None):
            _save_json(args.save_history, history)
        if getattr(args, "export_history_base", None):
            export_metrics(G, args.export_history_base, fmt=args.export_format)


def _to_float_array(values: Sequence[float] | None, *, name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence of numbers")
    return array


def _resolve_math_dimension(args: argparse.Namespace, fallback: int) -> int:
    dimension = getattr(args, "math_dimension", None)
    candidate_lengths: list[int] = []
    for attr in (
        "math_coherence_spectrum",
        "math_frequency_diagonal",
        "math_generator_diagonal",
    ):
        seq = getattr(args, attr, None)
        if seq is not None:
            candidate_lengths.append(len(seq))
    if dimension is None:
        if candidate_lengths:
            unique = set(candidate_lengths)
            if len(unique) > 1:
                raise ValueError("Math engine configuration requires matching sequence lengths")
            dimension = unique.pop()
        else:
            dimension = fallback
    else:
        for length in candidate_lengths:
            if length != dimension:
                raise ValueError("Math engine sequence lengths must match the requested dimension")
    if dimension is None or dimension <= 0:
        raise ValueError("Hilbert space dimension must be a positive integer")
    return int(dimension)


def _build_math_engine_config(G: "nx.Graph", args: argparse.Namespace) -> dict[str, Any]:
    node_count = _count_graph_nodes(G)
    fallback_dim = max(1, int(node_count) if node_count is not None else 1)
    dimension = _resolve_math_dimension(args, fallback=fallback_dim)

    coherence_spectrum = _to_float_array(
        getattr(args, "math_coherence_spectrum", None),
        name="--math-coherence-spectrum",
    )
    if coherence_spectrum is not None and coherence_spectrum.size != dimension:
        raise ValueError("Coherence spectrum length must equal the Hilbert dimension")

    frequency_diagonal = _to_float_array(
        getattr(args, "math_frequency_diagonal", None),
        name="--math-frequency-diagonal",
    )
    if frequency_diagonal is not None and frequency_diagonal.size != dimension:
        raise ValueError("Frequency diagonal length must equal the Hilbert dimension")

    generator_diagonal = _to_float_array(
        getattr(args, "math_generator_diagonal", None),
        name="--math-generator-diagonal",
    )
    if generator_diagonal is not None and generator_diagonal.size != dimension:
        raise ValueError("Generator diagonal length must equal the Hilbert dimension")

    coherence_c_min = getattr(args, "math_coherence_c_min", None)
    if coherence_spectrum is None:
        coherence_operator = make_coherence_operator(
            dimension,
            c_min=float(coherence_c_min) if coherence_c_min is not None else 0.1,
        )
    else:
        if coherence_c_min is not None:
            coherence_operator = CoherenceOperator(coherence_spectrum, c_min=float(coherence_c_min))
        else:
            coherence_operator = CoherenceOperator(coherence_spectrum)
        if not coherence_operator.is_positive_semidefinite():
            raise ValueError("Coherence spectrum must be positive semidefinite")

    frequency_matrix: np.ndarray
    if frequency_diagonal is None:
        frequency_matrix = np.eye(dimension, dtype=float)
    else:
        frequency_matrix = np.diag(frequency_diagonal)
    frequency_operator = make_frequency_operator(frequency_matrix)

    generator_matrix: np.ndarray
    if generator_diagonal is None:
        generator_matrix = np.zeros((dimension, dimension), dtype=float)
    else:
        generator_matrix = np.diag(generator_diagonal)

    hilbert_space = HilbertSpace(dimension)
    dynamics_engine = MathematicalDynamicsEngine(
        generator_matrix,
        hilbert_space=hilbert_space,
    )

    coherence_threshold = getattr(args, "math_coherence_threshold", None)
    if coherence_threshold is None:
        coherence_threshold = float(coherence_operator.c_min)
    else:
        coherence_threshold = float(coherence_threshold)

    state_projector = BasicStateProjector()
    validator = NFRValidator(
        hilbert_space,
        coherence_operator,
        coherence_threshold,
        frequency_operator=frequency_operator,
    )

    return {
        "enabled": True,
        "dimension": dimension,
        "hilbert_space": hilbert_space,
        "coherence_operator": coherence_operator,
        "frequency_operator": frequency_operator,
        "coherence_threshold": coherence_threshold,
        "state_projector": state_projector,
        "validator": validator,
        "dynamics_engine": dynamics_engine,
        "generator_matrix": generator_matrix,
    }


def _configure_math_engine(G: "nx.Graph", args: argparse.Namespace) -> None:
    if not getattr(args, "math_engine", False):
        G.graph.pop("MATH_ENGINE", None)
        return
    try:
        config = _build_math_engine_config(G, args)
    except ValueError as exc:
        logger.error("Math engine configuration error: %s", exc)
        raise SystemExit(1) from exc
    G.graph["MATH_ENGINE"] = config


def build_basic_graph(args: argparse.Namespace) -> "nx.Graph":
    """Construct the base graph topology described by CLI ``args``."""

    n = args.nodes
    topology = getattr(args, "topology", "ring").lower()
    seed = getattr(args, "seed", None)
    if topology == "ring":
        G = nx.cycle_graph(n)
    elif topology == "complete":
        G = nx.complete_graph(n)
    elif topology == "erdos":
        if getattr(args, "p", None) is not None:
            prob = float(args.p)
        else:
            if n <= 0:
                fallback = 0.0
            else:
                fallback = 3.0 / n
            prob = clamp01(fallback)
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"p must be between 0 and 1; received {prob}")
        G = nx.gnp_random_graph(n, prob, seed=seed)
    else:
        raise ValueError(
            f"Invalid topology '{topology}'. Accepted options are: ring, complete, erdos"
        )
    if seed is not None:
        G.graph["RANDOM_SEED"] = int(seed)
    return G


def apply_cli_config(G: "nx.Graph", args: argparse.Namespace) -> None:
    """Apply CLI overrides from ``args`` to graph-level configuration."""

    if args.config:
        try:
            apply_config(G, Path(args.config))
        except (StructuredFileError, ValueError) as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc
    arg_map = {
        "dt": ("DT", float),
        "integrator": ("INTEGRATOR_METHOD", str),
        "remesh_mode": ("REMESH_MODE", str),
        "glyph_hysteresis_window": ("GLYPH_HYSTERESIS_WINDOW", int),
    }
    for attr, (key, conv) in arg_map.items():
        val = getattr(args, attr, None)
        if val is not None:
            G.graph[key] = conv(val)

    base_gcanon: dict[str, Any]
    existing_gcanon = G.graph.get("GRAMMAR_CANON")
    if isinstance(existing_gcanon, Mapping):
        base_gcanon = {
            **METRIC_DEFAULTS["GRAMMAR_CANON"],
            **dict(existing_gcanon),
        }
    else:
        base_gcanon = dict(METRIC_DEFAULTS["GRAMMAR_CANON"])

    gcanon = {
        **base_gcanon,
        **_args_to_dict(args, prefix="grammar_"),
    }
    if getattr(args, "grammar_canon", None) is not None:
        gcanon["enabled"] = bool(args.grammar_canon)
    G.graph["GRAMMAR_CANON"] = gcanon

    selector = getattr(args, "selector", None)
    if selector is not None:
        sel_map = {
            "basic": default_glyph_selector,
            "param": parametric_glyph_selector,
        }
        G.graph["glyph_selector"] = sel_map.get(selector, default_glyph_selector)

    if hasattr(args, "gamma_type"):
        G.graph["GAMMA"] = {
            "type": args.gamma_type,
            "beta": args.gamma_beta,
            "R0": args.gamma_R0,
        }

    for attr, key in (
        ("trace_verbosity", "TRACE"),
        ("metrics_verbosity", "METRICS"),
    ):
        cfg = G.graph.get(key)
        if not isinstance(cfg, dict):
            cfg = deepcopy(METRIC_DEFAULTS[key])
            G.graph[key] = cfg
        value = getattr(args, attr, None)
        if value is not None:
            cfg["verbosity"] = value

    candidate_count = getattr(args, "um_candidate_count", None)
    if candidate_count is not None:
        G.graph["UM_CANDIDATE_COUNT"] = int(candidate_count)

    stop_window = getattr(args, "stop_early_window", None)
    stop_fraction = getattr(args, "stop_early_fraction", None)
    if stop_window is not None or stop_fraction is not None:
        stop_cfg = G.graph.get("STOP_EARLY")
        if isinstance(stop_cfg, Mapping):
            next_cfg = {**stop_cfg}
        else:
            next_cfg = deepcopy(METRIC_DEFAULTS["STOP_EARLY"])
        if stop_window is not None:
            next_cfg["window"] = int(stop_window)
        if stop_fraction is not None:
            next_cfg["fraction"] = float(stop_fraction)
        next_cfg.setdefault("enabled", True)
        G.graph["STOP_EARLY"] = next_cfg


def register_callbacks_and_observer(G: "nx.Graph") -> None:
    """Attach callbacks and validators required for CLI runs."""

    _attach_callbacks(G)
    validate_canon(G)


def _build_graph_from_args(args: argparse.Namespace) -> "nx.Graph":
    G = build_basic_graph(args)
    apply_cli_config(G, args)
    if getattr(args, "observer", False):
        G.graph["ATTACH_STD_OBSERVER"] = True
    prepare_network(G)
    register_callbacks_and_observer(G)
    _configure_math_engine(G, args)
    return G


def _load_sequence(path: Path) -> ProgramTokens:
    try:
        data = read_structured_file(path)
    except (StructuredFileError, OSError) as exc:
        if isinstance(exc, StructuredFileError):
            message = str(exc)
        else:
            message = str(StructuredFileError(path, exc))
        logger.error("%s", message)
        raise SystemExit(1) from exc
    if isinstance(data, Mapping) and "sequence" in data:
        data = data["sequence"]
    return parse_program_tokens(data)


def resolve_program(
    args: argparse.Namespace, default: Optional[ProgramTokens] = None
) -> Optional[ProgramTokens]:
    """Resolve preset/sequence inputs into program tokens."""

    if getattr(args, "preset", None):
        try:
            return get_preset(args.preset)
        except KeyError as exc:
            details = exc.args[0] if exc.args else "Preset lookup failed."
            logger.error(
                (
                    "Unknown preset '%s'. Available presets: %s. %s "
                    "Use --sequence-file to execute custom sequences."
                ),
                args.preset,
                _PREFERRED_PRESETS_DISPLAY,
                details,
            )
            raise SystemExit(1) from exc
    if getattr(args, "sequence_file", None):
        return _load_sequence(Path(args.sequence_file))
    return default


def run_program(
    G: Optional["nx.Graph"],
    program: Optional[ProgramTokens],
    args: argparse.Namespace,
) -> "nx.Graph":
    """Execute ``program`` (or timed run) on ``G`` using CLI options."""

    if G is None:
        G = _build_graph_from_args(args)

    if program is None:
        steps = getattr(args, "steps", 100)
        steps = 100 if steps is None else int(steps)
        if steps < 0:
            steps = 0

        run_kwargs: dict[str, Any] = {}
        for attr in ("dt", "use_Si", "apply_glyphs"):
            value = getattr(args, attr, None)
            if value is not None:
                run_kwargs[attr] = value

        job_overrides: dict[str, Any] = {}
        dnfr_jobs = getattr(args, "dnfr_n_jobs", None)
        if dnfr_jobs is not None:
            job_overrides["dnfr_n_jobs"] = int(dnfr_jobs)
        if job_overrides:
            run_kwargs["n_jobs"] = job_overrides

        run(G, steps=steps, **run_kwargs)
    else:
        play(G, program)

    _persist_history(G, args)
    return G


def _run_cli_program(
    args: argparse.Namespace,
    *,
    default_program: Optional[ProgramTokens] = None,
    graph: Optional["nx.Graph"] = None,
) -> tuple[int, Optional["nx.Graph"]]:
    try:
        program = resolve_program(args, default=default_program)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code or 1, None

    try:
        result_graph = run_program(graph, program, args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code or 1, None
    return 0, result_graph


def _log_math_engine_summary(G: "nx.Graph") -> None:
    math_cfg = G.graph.get("MATH_ENGINE")
    if not isinstance(math_cfg, Mapping) or not math_cfg.get("enabled"):
        return

    nodes = list(G.nodes)
    if not nodes:
        logger.info("[MATH] Math engine validation skipped: no nodes present")
        return

    hilbert_space: HilbertSpace = math_cfg["hilbert_space"]
    coherence_operator: CoherenceOperator = math_cfg["coherence_operator"]
    frequency_operator: FrequencyOperator | None = math_cfg.get("frequency_operator")
    state_projector: BasicStateProjector = math_cfg.get("state_projector", BasicStateProjector())
    validator: NFRValidator | None = math_cfg.get("validator")
    if validator is None:
        coherence_threshold = math_cfg.get("coherence_threshold")
        validator = NFRValidator(
            hilbert_space,
            coherence_operator,
            float(coherence_threshold) if coherence_threshold is not None else 0.0,
            frequency_operator=frequency_operator,
        )
        math_cfg["validator"] = validator

    enforce_frequency = bool(frequency_operator is not None)

    norm_values: list[float] = []
    normalized_flags: list[bool] = []
    coherence_flags: list[bool] = []
    coherence_values: list[float] = []
    coherence_threshold: float | None = None
    frequency_flags: list[bool] = []
    frequency_values: list[float] = []
    frequency_spectrum_min: float | None = None

    for node_id in nodes:
        data = G.nodes[node_id]
        epi = float(
            get_attr(
                data,
                EPI_ALIAS_KEYS,
                default=0.0,
            )
        )
        nu_f = float(
            get_attr(
                data,
                VF_ALIAS_KEYS,
                default=float(data.get(VF_PRIMARY, 0.0)),
            )
        )
        theta = float(data.get("theta", 0.0))
        state = state_projector(epi=epi, nu_f=nu_f, theta=theta, dim=hilbert_space.dimension)
        norm_values.append(float(hilbert_space.norm(state)))
        outcome = validator.validate(
            state,
            enforce_frequency_positivity=enforce_frequency,
        )
        summary = outcome.summary
        normalized_flags.append(bool(summary.get("normalized", False)))

        coherence_summary = summary.get("coherence")
        if isinstance(coherence_summary, Mapping):
            coherence_flags.append(bool(coherence_summary.get("passed", False)))
            coherence_values.append(float(coherence_summary.get("value", 0.0)))
            if coherence_threshold is None and "threshold" in coherence_summary:
                coherence_threshold = float(coherence_summary.get("threshold", 0.0))

        frequency_summary = summary.get("frequency")
        if isinstance(frequency_summary, Mapping):
            frequency_flags.append(bool(frequency_summary.get("passed", False)))
            frequency_values.append(float(frequency_summary.get("value", 0.0)))
            if frequency_spectrum_min is None and "spectrum_min" in frequency_summary:
                frequency_spectrum_min = float(frequency_summary.get("spectrum_min", 0.0))

    if norm_values:
        logger.info(
            "[MATH] Hilbert norm preserved=%s (min=%.6f, max=%.6f)",
            all(normalized_flags),
            min(norm_values),
            max(norm_values),
        )

    if coherence_values and coherence_threshold is not None:
        logger.info(
            "[MATH] Coherence ≥ C_min=%s (C_min=%.6f, min=%.6f)",
            all(coherence_flags),
            float(coherence_threshold),
            min(coherence_values),
        )

    if frequency_values:
        if frequency_spectrum_min is not None:
            logger.info(
                "[MATH] νf positivity=%s (min=%.6f, spectrum_min=%.6f)",
                all(frequency_flags),
                min(frequency_values),
                frequency_spectrum_min,
            )
        else:
            logger.info(
                "[MATH] νf positivity=%s (min=%.6f)",
                all(frequency_flags),
                min(frequency_values),
            )


def _log_run_summaries(G: "nx.Graph", args: argparse.Namespace) -> None:
    cfg_coh = G.graph.get("COHERENCE", METRIC_DEFAULTS["COHERENCE"])
    cfg_diag = G.graph.get("DIAGNOSIS", METRIC_DEFAULTS["DIAGNOSIS"])
    hist = ensure_history(G)

    if cfg_coh.get("enabled", True):
        Wstats = hist.get(cfg_coh.get("stats_history_key", "W_stats"), [])
        if Wstats:
            logger.info("[COHERENCE] last step: %s", Wstats[-1])

    if cfg_diag.get("enabled", True):
        last_diag = hist.get(cfg_diag.get("history_key", "nodal_diag"), [])
        if last_diag:
            sample = list(last_diag[-1].values())[:3]
            logger.info("[DIAGNOSIS] sample: %s", sample)

    if args.summary:
        summary_limit = getattr(args, "summary_limit", DEFAULT_SUMMARY_SERIES_LIMIT)
        summary, has_latency_values = build_metrics_summary(G, series_limit=summary_limit)
        logger.info("Global Tg: %s", summary["Tg_global"])
        logger.info("Top operators by Tg: %s", glyph_top(G, k=5))
        if has_latency_values:
            logger.info("Average latency: %s", summary["latency_mean"])

    _log_math_engine_summary(G)


def cmd_run(args: argparse.Namespace) -> int:
    """Execute ``tnfr run`` returning the exit status."""

    code, graph = _run_cli_program(args)
    if code != 0:
        return code

    if graph is not None:
        _log_run_summaries(graph, args)
    return 0


def cmd_sequence(args: argparse.Namespace) -> int:
    """Execute ``tnfr sequence`` returning the exit status."""

    if args.preset and args.sequence_file:
        logger.error("Cannot use --preset and --sequence-file at the same time")
        return 1
    code, _ = _run_cli_program(args, default_program=get_preset(CANONICAL_PRESET_NAME))
    return code


def cmd_metrics(args: argparse.Namespace) -> int:
    """Execute ``tnfr metrics`` returning the exit status."""

    if getattr(args, "steps", None) is None:
        # Default a longer run for metrics stability
        args.steps = 200

    code, graph = _run_cli_program(args)
    if code != 0 or graph is None:
        return code

    summary_limit = getattr(args, "summary_limit", None)
    out, _ = build_metrics_summary(graph, series_limit=summary_limit)
    if args.save:
        _save_json(args.save, out)
    else:
        logger.info("%s", json_dumps(out))
    return 0


def cmd_profile_si(args: argparse.Namespace) -> int:
    """Execute ``tnfr profile-si`` returning the exit status."""

    try:
        profile_module = import_module("benchmarks.compute_si_profile")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        logger.error("Sense Index profiling helpers unavailable: %s", exc)
        return 1

    profile_compute_si = getattr(profile_module, "profile_compute_si")

    profile_compute_si(
        node_count=int(args.nodes),
        chord_step=int(args.chord_step),
        loops=int(args.loops),
        output_dir=Path(args.output_dir),
        fmt=str(args.format),
        sort=str(args.sort),
    )
    return 0


def cmd_profile_pipeline(args: argparse.Namespace) -> int:
    """Execute ``tnfr profile-pipeline`` returning the exit status."""

    try:
        profile_module = import_module("benchmarks.full_pipeline_profile")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        logger.error("Full pipeline profiling helpers unavailable: %s", exc)
        return 1

    profile_full_pipeline = getattr(profile_module, "profile_full_pipeline")

    try:
        si_chunk_sizes = _parse_cli_variants(getattr(args, "si_chunk_sizes", None))
        dnfr_chunk_sizes = _parse_cli_variants(getattr(args, "dnfr_chunk_sizes", None))
        si_workers = _parse_cli_variants(getattr(args, "si_workers", None))
        dnfr_workers = _parse_cli_variants(getattr(args, "dnfr_workers", None))
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    profile_full_pipeline(
        node_count=int(args.nodes),
        edge_probability=float(args.edge_probability),
        loops=int(args.loops),
        seed=int(args.seed),
        output_dir=Path(args.output_dir),
        sort=str(args.sort),
        si_chunk_sizes=si_chunk_sizes,
        dnfr_chunk_sizes=dnfr_chunk_sizes,
        si_workers=si_workers,
        dnfr_workers=dnfr_workers,
    )
    return 0


def cmd_math_run(args: argparse.Namespace) -> int:
    """Execute ``tnfr math.run`` returning the exit status.

    This command always enables the mathematical dynamics engine for
    validation of TNFR structural invariants on Hilbert space.
    """

    # Force math engine to be enabled
    setattr(args, "math_engine", True)

    # Set default attributes if not present
    if not hasattr(args, "summary"):
        setattr(args, "summary", False)
    if not hasattr(args, "summary_limit"):
        setattr(args, "summary_limit", DEFAULT_SUMMARY_SERIES_LIMIT)

    code, graph = _run_cli_program(args)
    if code != 0:
        return code

    if graph is not None:
        _log_run_summaries(graph, args)
        logger.info("[MATH.RUN] Mathematical dynamics validation completed")
    return 0


def cmd_epi_validate(args: argparse.Namespace) -> int:
    """Execute ``tnfr epi.validate`` returning the exit status.

    This command validates EPI structural integrity, coherence preservation,
    and operator closure according to TNFR canonical invariants.
    """

    code, graph = _run_cli_program(args)
    if code != 0:
        return code

    if graph is None:
        logger.error("[EPI.VALIDATE] No graph generated for validation")
        return 1

    # Validation checks
    tolerance = getattr(args, "tolerance", 1e-6)
    check_coherence = getattr(args, "check_coherence", True)
    check_frequency = getattr(args, "check_frequency", True)
    check_phase = getattr(args, "check_phase", True)

    validation_passed = True
    validation_summary = []

    # Check coherence preservation
    if check_coherence:
        hist = ensure_history(graph)
        cfg_coh = graph.graph.get("COHERENCE", METRIC_DEFAULTS["COHERENCE"])
        if cfg_coh.get("enabled", True):
            Wstats = hist.get(cfg_coh.get("stats_history_key", "W_stats"), [])
            if Wstats:
                # Check that coherence is non-negative and bounded
                for i, stats in enumerate(Wstats):
                    W_mean = float(stats.get("mean", 0.0))
                    if W_mean < -tolerance:
                        validation_passed = False
                        validation_summary.append(
                            f"  [FAIL] Step {i}: Coherence W_mean={W_mean:.6f} < 0"
                        )
                if validation_passed:
                    validation_summary.append(
                        f"  [PASS] Coherence preserved (W_mean ≥ 0 across {len(Wstats)} steps)"
                    )
            else:
                validation_summary.append("  [SKIP] No coherence history available")
        else:
            validation_summary.append("  [SKIP] Coherence tracking disabled")

    # Check structural frequency positivity
    if check_frequency:
        nodes = list(_iter_graph_nodes(graph))
        if nodes:
            negative_frequencies = []
            for node_id in nodes:
                data = graph.nodes[node_id]
                nu_f = float(
                    get_attr(
                        data,
                        VF_ALIAS_KEYS,
                        default=float(data.get(VF_PRIMARY, 0.0)),
                    )
                )
                if nu_f < -tolerance:
                    negative_frequencies.append((node_id, nu_f))

            if negative_frequencies:
                validation_passed = False
                for node_id, nu_f in negative_frequencies[:5]:  # Show first 5
                    validation_summary.append(f"  [FAIL] Node {node_id}: νf={nu_f:.6f} < 0")
                if len(negative_frequencies) > 5:
                    validation_summary.append(
                        f"  ... and {len(negative_frequencies) - 5} more nodes"
                    )
            else:
                validation_summary.append(
                    f"  [PASS] Structural frequency νf ≥ 0 for all {len(nodes)} nodes"
                )
        else:
            validation_summary.append("  [SKIP] No nodes to validate")

    # Check phase synchrony in couplings
    if check_phase:
        edges = list(_iter_graph_edges(graph))
        if edges:
            phase_violations = []
            for u, v in edges:
                theta_u = float(graph.nodes[u].get("theta", 0.0))
                theta_v = float(graph.nodes[v].get("theta", 0.0))
                # Check if phases are defined (not both zero)
                if abs(theta_u) > tolerance or abs(theta_v) > tolerance:
                    # Phase difference should be bounded
                    phase_diff = abs(theta_u - theta_v)
                    if phase_diff > TWO_PI:  # > 2π
                        phase_violations.append((u, v, phase_diff))

            if phase_violations:
                validation_passed = False
                for u, v, diff in phase_violations[:5]:
                    validation_summary.append(
                        f"  [WARN] Edge ({u},{v}): phase diff={diff:.6f} > 2π"
                    )
                if len(phase_violations) > 5:
                    validation_summary.append(f"  ... and {len(phase_violations) - 5} more edges")
            else:
                validation_summary.append(
                    f"  [PASS] Phase synchrony maintained across {len(edges)} edges"
                )
        else:
            validation_summary.append("  [SKIP] No edges to validate")

    # Log validation results
    logger.info("[EPI.VALIDATE] Validation Summary:")
    for line in validation_summary:
        logger.info("%s", line)

    if validation_passed:
        logger.info("[EPI.VALIDATE] ✓ All validation checks passed")
        return 0
    else:
        logger.info("[EPI.VALIDATE] ✗ Some validation checks failed")
        return 1
