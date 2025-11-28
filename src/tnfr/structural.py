"""Maintain TNFR structural coherence for nodes and operator sequences.

This module exposes the canonical entry points used by the engine to
instantiate coherent TNFR nodes and to orchestrate structural operator
pipelines while keeping the nodal equation
``∂EPI/∂t = νf · ΔNFR(t)`` balanced.  Consumers are expected to provide
graph instances honouring :class:`tnfr.types.GraphLike`: the structural
helpers reach into ``nodes``, ``neighbors``, ``number_of_nodes`` and the
``.graph`` metadata mapping to propagate ΔNFR hooks and coherence metrics.

Public API
----------
create_nfr
    Initialise a node with canonical EPI, νf and phase attributes plus a
    ΔNFR hook that propagates reorganisations through the graph.
run_sequence
    Validate and execute operator trajectories so that ΔNFR hooks can
    update EPI, νf and phase coherently after each step.
OPERATORS
    Registry of canonical structural operators ready to be composed into
    validated sequences.
validate_sequence
    Grammar guard that ensures operator trajectories stay within TNFR
    closure rules before execution.

CRITICAL TECHNICAL NOTE (Context Override for Pre-existing EPI - Nov 2025):
---------------------------------------------------------------------------
run_sequence() automatically detects when a node has pre-existing EPI (EPI ≠ 0)
and passes context={'initial_epi_nonzero': True} to validate_sequence().

This bypasses the U1a generator requirement, which states that sequences
must start with a generator {AL, NAV, REMESH} when EPI=0.

**Physics Rationale**:
- U1a exists because ∂EPI/∂t is undefined at EPI=0 (no structure to evolve)
- When EPI already exists, the node has structure that CAN evolve
- Applying operators like Coherence, Resonance, etc. is physically valid
- Strict U1a enforcement would prevent legitimate operations on existing nodes

**Implementation**: check_epi_nonzero() examines node EPI attribute:
  - If EPI ≠ 0: auto-pass context override to grammar validator
  - If EPI = 0: strict U1a enforcement (must start with generator)

**Canonicity Preserved**: This does NOT weaken grammar. U1a's physical
basis (undefined evolution at EPI=0) does not apply when structure exists.
The override is a necessary operational flexibility, not a violation.

See: run_sequence() context detection, grammar_patterns._check_start_rule()
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Mapping, Sequence, cast

import networkx as nx

from .constants import EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from .dynamics import (
    dnfr_epi_vf_mixed,
    set_delta_nfr_hook,
)
from .mathematics import (
    BasicStateProjector,
    CoherenceOperator,
    FrequencyOperator,
    HilbertSpace,
    MathematicalDynamicsEngine,
    make_coherence_operator,
    make_frequency_operator,
)
from tnfr.validation import (
    NFRValidator,
    validate_sequence,
    TNFRValidator as InvariantValidator,
    SequenceSemanticValidator,
    InvariantSeverity,
    validation_config,
)
from tnfr.config.operator_names import VALID_START_OPERATORS
from .operators.definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Operator,
    Reception,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
)
from .operators.registry import OPERATORS
from .types import DeltaNFRHook, NodeId, TNFRGraph
from .utils import get_logger

try:  # pragma: no cover - optional dependency path exercised in CI extras
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency path exercised in CI extras
    np = None  # type: ignore[assignment]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 1) NFR factory
# ---------------------------------------------------------------------------


def create_nfr(
    name: str,
    *,
    epi: float = 0.0,
    vf: float = 1.0,
    theta: float = 0.0,
    graph: TNFRGraph | None = None,
    dnfr_hook: DeltaNFRHook = dnfr_epi_vf_mixed,
) -> tuple[TNFRGraph, str]:
    """Anchor a TNFR node by seeding EPI, νf, phase and ΔNFR coupling.

    The factory secures the structural state of a node: it stores canonical
    values for the Primary Information Structure (EPI), structural frequency
    (νf) and phase, then installs a ΔNFR hook so that later operator
    sequences can reorganise the node without breaking the nodal equation.

    Parameters
    ----------
    name : str
        Identifier for the new node. The identifier is stored as the node key
        and must remain hashable by :mod:`networkx`.
    epi : float, optional
        Initial Primary Information Structure (EPI) assigned to the node. The
        value provides the baseline form that subsequent ΔNFR hooks reorganise
        through the nodal equation.
    vf : float, optional
        Structural frequency (νf, expressed in Hz_str) used as the starting
        reorganisation rate for the node.
    theta : float, optional
        Initial phase of the node in radians, used to keep phase alignment with
        neighbouring coherence structures.
    graph : TNFRGraph, optional
        Existing graph where the node will be registered. When omitted a new
        :class:`networkx.Graph` instance is created.
    dnfr_hook : DeltaNFRHook, optional
        Callable responsible for computing ΔNFR and updating EPI/νf after each
        operator application. By default the canonical ``dnfr_epi_vf_mixed``
        hook is installed, which keeps the nodal equation coherent with TNFR
        invariants.

    Returns
    -------
    tuple[TNFRGraph, str]
        The graph that stores the node together with the node identifier. The
        tuple form allows immediate reuse with :func:`run_sequence`.

    Notes
    -----
    The factory does not introduce additional TNFR-specific errors. Any
    exceptions raised by :mod:`networkx` when adding nodes propagate unchanged.

    Examples
    --------
    Create a node, connect a ΔNFR hook and launch a coherent operator
    trajectory while tracking the evolving metrics.

    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import (
    ...     Coherence,
    ...     Emission,
    ...     Reception,
    ...     Resonance,
    ...     Silence,
    ...     create_nfr,
    ...     run_sequence,
    ... )
    >>> G, node = create_nfr("seed", epi=1.0, vf=2.0, theta=0.1)
    >>> def synchronise_delta(graph):
    ...     delta = graph.nodes[node][VF_PRIMARY] * 0.2
    ...     graph.nodes[node][DNFR_PRIMARY] = delta
    ...     graph.nodes[node][EPI_PRIMARY] += delta
    ...     graph.nodes[node][VF_PRIMARY] += delta * 0.05
    ...     graph.nodes[node][THETA_PRIMARY] += 0.01
    >>> set_delta_nfr_hook(G, synchronise_delta)
    >>> run_sequence(G, node, [Emission(), Reception(), Coherence(), Resonance(), Silence()])  # doctest: +SKIP
    >>> (
    ...     G.nodes[node][EPI_PRIMARY],
    ...     G.nodes[node][VF_PRIMARY],
    ...     G.nodes[node][THETA_PRIMARY],
    ...     G.nodes[node][DNFR_PRIMARY],
    ... )  # doctest: +SKIP
    (..., ..., ..., ...)
    """
    from .validation.input_validation import (
        ValidationError,
        validate_epi_value,
        validate_node_id,
        validate_theta_value,
        validate_tnfr_graph,
        validate_vf_value,
    )

    # Validate input parameters
    try:
        validate_node_id(name)
        config = graph.graph if graph is not None else None
        epi = validate_epi_value(epi, config=config, allow_complex=False)
        vf = validate_vf_value(vf, config=config)
        theta = validate_theta_value(theta, normalize=False)
        if graph is not None:
            validate_tnfr_graph(graph)
    except ValidationError as e:
        raise ValueError(f"Invalid parameters for create_nfr: {e}") from e

    G = graph if graph is not None else nx.Graph()
    G.add_node(
        name,
        **{
            EPI_PRIMARY: float(epi),
            VF_PRIMARY: float(vf),
            THETA_PRIMARY: float(theta),
        },
    )
    set_delta_nfr_hook(G, dnfr_hook)
    return G, name


def _resolve_dimension(
    G: TNFRGraph,
    *,
    dimension: int | None,
    hilbert_space: HilbertSpace | None,
    existing_cfg: Mapping[str, object] | None,
) -> int:
    if hilbert_space is not None:
        resolved = int(getattr(hilbert_space, "dimension", 0) or 0)
        if resolved <= 0:
            raise ValueError("Hilbert space dimension must be positive.")
        return resolved

    if dimension is None and existing_cfg:
        candidate = existing_cfg.get("dimension")
        if isinstance(candidate, int) and candidate > 0:
            dimension = candidate

    if dimension is None:
        if hasattr(G, "number_of_nodes"):
            count = int(G.number_of_nodes())
        else:
            count = len(tuple(G.nodes))
        dimension = max(1, count)

    resolved = int(dimension)
    if resolved <= 0:
        raise ValueError("Hilbert space dimension must be positive.")
    return resolved


def _ensure_coherence_operator(
    *,
    operator: CoherenceOperator | None,
    dimension: int,
    spectrum: Sequence[float] | None,
    c_min: float | None,
) -> CoherenceOperator:
    if operator is not None:
        return operator

    kwargs: dict[str, object] = {}
    if spectrum is not None:
        spectrum_array = np.asarray(spectrum, dtype=np.complex128)
        if spectrum_array.ndim != 1:
            raise ValueError("Coherence spectrum must be one-dimensional.")
        kwargs["spectrum"] = spectrum_array
    if c_min is not None:
        kwargs["c_min"] = float(c_min)
    return make_coherence_operator(dimension, **kwargs)


def _ensure_frequency_operator(
    *,
    operator: FrequencyOperator | None,
    dimension: int,
    diagonal: Sequence[float] | None,
) -> FrequencyOperator:
    if operator is not None:
        return operator

    if diagonal is None:
        matrix = np.eye(dimension, dtype=float)
    else:
        diag_array = np.asarray(diagonal, dtype=float)
        if diag_array.ndim != 1:
            raise ValueError("Frequency diagonal must be one-dimensional.")
        if diag_array.shape[0] != int(dimension):
            raise ValueError("Frequency diagonal size must match Hilbert dimension.")
        matrix = np.diag(diag_array)
    return make_frequency_operator(np.asarray(matrix, dtype=np.complex128))


def _ensure_generator_matrix(
    *,
    dimension: int,
    diagonal: Sequence[float] | None,
) -> "np.ndarray":
    if diagonal is None:
        return np.zeros((dimension, dimension), dtype=np.complex128)
    diag_array = np.asarray(diagonal, dtype=np.complex128)
    if diag_array.ndim != 1:
        raise ValueError("Generator diagonal must be one-dimensional.")
    if diag_array.shape[0] != int(dimension):
        raise ValueError("Generator diagonal size must match Hilbert dimension.")
    return np.diag(diag_array)


def create_math_nfr(
    name: str,
    *,
    epi: float = 0.0,
    vf: float = 1.0,
    theta: float = 0.0,
    graph: TNFRGraph | None = None,
    dnfr_hook: DeltaNFRHook = dnfr_epi_vf_mixed,
    dimension: int | None = None,
    hilbert_space: HilbertSpace | None = None,
    coherence_operator: CoherenceOperator | None = None,
    coherence_spectrum: Sequence[float] | None = None,
    coherence_c_min: float | None = None,
    coherence_threshold: float | None = None,
    frequency_operator: FrequencyOperator | None = None,
    frequency_diagonal: Sequence[float] | None = None,
    generator_diagonal: Sequence[float] | None = None,
    state_projector: BasicStateProjector | None = None,
    dynamics_engine: MathematicalDynamicsEngine | None = None,
    validator: NFRValidator | None = None,
) -> tuple[TNFRGraph, str]:
    """Create a TNFR node with canonical mathematical validation attached.

    The helper wraps :func:`create_nfr` while projecting the structural state
    into a Hilbert space so coherence, νf and norm invariants can be tracked via
    the mathematical runtime.  It installs operators and validation metadata on
    both the node and the hosting graph so that the
    :class:`~tnfr.mathematics.MathematicalDynamicsEngine` can consume them
    directly.

    Parameters
    ----------
    name : str
        Identifier for the new node.
    epi, vf, theta : float, optional
        Canonical TNFR scalars forwarded to :func:`create_nfr`.
    dimension : int, optional
        Hilbert space dimension. When omitted it is inferred from the graph size
        (at least one).
    hilbert_space : HilbertSpace, optional
        Pre-built Hilbert space to reuse. Its dimension supersedes ``dimension``.
    coherence_operator, frequency_operator : optional
        Custom operators to install. When omitted they are derived from
        ``coherence_spectrum``/``coherence_c_min`` and
        ``frequency_diagonal`` respectively.
    coherence_threshold : float, optional
        Validation floor. Defaults to ``coherence_operator.c_min``.
    generator_diagonal : sequence of float, optional
        Diagonal entries for the unitary generator used by the mathematical
        dynamics engine. Defaults to a null generator.
    state_projector : BasicStateProjector, optional
        Projector used to build the canonical spectral state for validation.

    Returns
    -------
    tuple[TNFRGraph, str]
        The graph and node identifier, mirroring :func:`create_nfr`.

    Examples
    --------
    >>> G, node = create_math_nfr("math-seed", epi=0.4, vf=1.2, theta=0.05, dimension=3)
    >>> metrics = G.nodes[node]["math_metrics"]
    >>> round(metrics["norm"], 6)
    1.0
    >>> metrics["coherence_passed"], metrics["frequency_passed"]
    (True, True)
    >>> metrics["coherence_value"] >= metrics["coherence_threshold"]
    True

    Notes
    -----
    The helper mutates/extends ``G.graph['MATH_ENGINE']`` so subsequent calls to
    :mod:`tnfr.dynamics.runtime` can advance the mathematical engine without
    further configuration.
    """

    if np is None:
        raise ImportError("create_math_nfr requires NumPy; install the 'tnfr[math]' extras.")

    G, node = create_nfr(
        name,
        epi=epi,
        vf=vf,
        theta=theta,
        graph=graph,
        dnfr_hook=dnfr_hook,
    )

    existing_cfg = G.graph.get("MATH_ENGINE")
    mapping_cfg: Mapping[str, object] | None
    if isinstance(existing_cfg, Mapping):
        mapping_cfg = existing_cfg
    else:
        mapping_cfg = None

    resolved_dimension = _resolve_dimension(
        G,
        dimension=dimension,
        hilbert_space=hilbert_space,
        existing_cfg=mapping_cfg,
    )

    hilbert = hilbert_space or HilbertSpace(resolved_dimension)
    resolved_dimension = int(getattr(hilbert, "dimension", resolved_dimension))

    coherence = _ensure_coherence_operator(
        operator=coherence_operator,
        dimension=resolved_dimension,
        spectrum=coherence_spectrum,
        c_min=coherence_c_min,
    )
    threshold = float(coherence_threshold if coherence_threshold is not None else coherence.c_min)

    frequency = _ensure_frequency_operator(
        operator=frequency_operator,
        dimension=resolved_dimension,
        diagonal=frequency_diagonal,
    )

    projector = state_projector or BasicStateProjector()

    generator_matrix = _ensure_generator_matrix(
        dimension=resolved_dimension,
        diagonal=generator_diagonal,
    )
    engine = dynamics_engine or MathematicalDynamicsEngine(
        generator_matrix,
        hilbert_space=hilbert,
    )

    enforce_frequency = frequency is not None
    spectral_validator = validator or NFRValidator(
        hilbert,
        coherence,
        threshold,
        frequency_operator=frequency if enforce_frequency else None,
    )

    state = projector(
        epi=float(epi),
        nu_f=float(vf),
        theta=float(theta),
        dim=resolved_dimension,
    )
    norm_value = float(hilbert.norm(state))
    outcome = spectral_validator.validate(
        state,
        enforce_frequency_positivity=enforce_frequency,
    )
    summary_raw = outcome.summary
    summary = {key: deepcopy(value) for key, value in summary_raw.items()}

    coherence_summary = summary.get("coherence")
    frequency_summary = summary.get("frequency")

    math_metrics = {
        "norm": norm_value,
        "normalized": bool(summary.get("normalized", False)),
        "coherence_value": (
            float(coherence_summary.get("value", 0.0))
            if isinstance(coherence_summary, Mapping)
            else 0.0
        ),
        "coherence_threshold": (
            float(coherence_summary.get("threshold", threshold))
            if isinstance(coherence_summary, Mapping)
            else threshold
        ),
        "coherence_passed": (
            bool(coherence_summary.get("passed", False))
            if isinstance(coherence_summary, Mapping)
            else False
        ),
        "frequency_value": (
            float(frequency_summary.get("value", 0.0))
            if isinstance(frequency_summary, Mapping)
            else 0.0
        ),
        "frequency_passed": (
            bool(frequency_summary.get("passed", False))
            if isinstance(frequency_summary, Mapping)
            else True
        ),
        "frequency_spectrum_min": (
            float(frequency_summary.get("spectrum_min", 0.0))
            if isinstance(frequency_summary, Mapping) and "spectrum_min" in frequency_summary
            else None
        ),
        "unitary_passed": bool(summary.get("unitary_stability", {}).get("passed", False)),
    }

    node_context = {
        "hilbert_space": hilbert,
        "coherence_operator": coherence,
        "frequency_operator": frequency,
        "coherence_threshold": threshold,
        "dimension": resolved_dimension,
    }

    node_data = G.nodes[node]
    node_data["math_metrics"] = math_metrics
    node_data["math_summary"] = summary
    node_data["math_context"] = node_context

    cfg = dict(mapping_cfg) if mapping_cfg is not None else {}
    cfg.update(
        {
            "enabled": True,
            "dimension": resolved_dimension,
            "hilbert_space": hilbert,
            "coherence_operator": coherence,
            "coherence_threshold": threshold,
            "frequency_operator": frequency,
            "state_projector": projector,
            "validator": spectral_validator,
            "generator_matrix": generator_matrix,
            "dynamics_engine": engine,
        }
    )
    G.graph["MATH_ENGINE"] = cfg

    return G, node


__all__ = (
    "create_nfr",
    "create_math_nfr",
    "Operator",
    "Emission",
    "Reception",
    "Coherence",
    "Dissonance",
    "Coupling",
    "Resonance",
    "Silence",
    "Expansion",
    "Contraction",
    "SelfOrganization",
    "Mutation",
    "Transition",
    "Recursivity",
    "OPERATORS",
    "validate_sequence",
    "run_sequence",
)


def run_sequence(G: TNFRGraph, node: NodeId, ops: Iterable[Operator]) -> None:
    """Drive structural sequences that rebalance EPI, νf, phase and ΔNFR.

    The function enforces the canonical operator grammar, then executes each
    operator so that the configured ΔNFR hook can update the nodal equation in
    place. Each step is expected to express the structural effect of the
    operator, while the hook keeps EPI, νf and phase consistent with the
    resulting ΔNFR variations.

    Parameters
    ----------
    G : TNFRGraph
        Graph that stores the node and its ΔNFR orchestration hook. The hook is
        read from ``G.graph['compute_delta_nfr']`` and is responsible for
        keeping the nodal equation up to date after each operator.
    node : NodeId
        Identifier of the node that will receive the operators. The node must
        already contain the canonical attributes ``EPI``, ``νf`` and ``θ``.
    ops : Iterable[Operator]
        Iterable of canonical structural operators to apply. Their
        concatenation must respect the validated TNFR grammar.

    Returns
    -------
    None
        The function mutates ``G`` in-place by updating the node attributes.

    Raises
    ------
    ValueError
        Raised when the provided operator names do not satisfy the canonical
        sequence validation rules.

    Examples
    --------
    Run a validated trajectory that highlights the ΔNFR-driven evolution of the
    node metrics.

    >>> from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import (
    ...     Coherence,
    ...     Emission,
    ...     Reception,
    ...     Resonance,
    ...     Silence,
    ...     create_nfr,
    ...     run_sequence,
    ... )
    >>> G, node = create_nfr("seed", epi=0.8, vf=1.5, theta=0.0)
    >>> def amplify_delta(graph):
    ...     delta = graph.nodes[node][VF_PRIMARY] * 0.15
    ...     graph.nodes[node][DNFR_PRIMARY] = delta
    ...     graph.nodes[node][EPI_PRIMARY] += delta * 0.8
    ...     graph.nodes[node][VF_PRIMARY] += delta * 0.1
    ...     graph.nodes[node][THETA_PRIMARY] += 0.02
    >>> set_delta_nfr_hook(G, amplify_delta)
    >>> run_sequence(G, node, [Emission(), Reception(), Coherence(), Resonance(), Silence()])  # doctest: +SKIP
    >>> (
    ...     G.nodes[node][EPI_PRIMARY],
    ...     G.nodes[node][VF_PRIMARY],
    ...     G.nodes[node][THETA_PRIMARY],
    ...     G.nodes[node][DNFR_PRIMARY],
    ... )  # doctest: +SKIP
    (..., ..., ..., ...)
    """

    compute = G.graph.get("compute_delta_nfr")
    ops_list = list(ops)
    names = [op.name for op in ops_list]

    # Initialize validators (reuse global instances for performance)
    if not hasattr(run_sequence, "_invariant_validator"):
        run_sequence._invariant_validator = InvariantValidator()  # type: ignore[attr-defined]
    if not hasattr(run_sequence, "_semantic_validator"):
        run_sequence._semantic_validator = SequenceSemanticValidator()  # type: ignore[attr-defined]

    # Skip validation for empty sequences (TNFR: empty sequence is structural identity)
    if names:
        # Birth context detection: if node already has non-zero EPI and
        # sequence begins with a non-generator we allow context override.
        epi_val = G.nodes[node].get(EPI_PRIMARY, 0.0)
        if names[0] not in VALID_START_OPERATORS and epi_val:
            logger.info(
                "U1a override (EPI≠0) node=%s; start=%s",
                node,
                names[0],
            )
            outcome = validate_sequence(
                names, context={"initial_epi_nonzero": True}
            )
        else:
            outcome = validate_sequence(names)
        if not outcome.passed:
            summary_message = outcome.summary.get("message", "validation failed")
            raise ValueError(f"Invalid sequence: {summary_message}")

        # Semantic validation of sequence (if enabled)
        if validation_config.enable_semantic_validation:
            semantic_violations = run_sequence._semantic_validator.validate_semantic_sequence(names)  # type: ignore[attr-defined]
            if semantic_violations:
                # Filter by configured settings
                error_violations = [
                    v
                    for v in semantic_violations
                    if v.severity == InvariantSeverity.ERROR
                    or v.severity == InvariantSeverity.CRITICAL
                ]
                warning_violations = [
                    v for v in semantic_violations if v.severity == InvariantSeverity.WARNING
                ]

                # Always raise on errors
                if error_violations:
                    report = run_sequence._invariant_validator.generate_report(error_violations)  # type: ignore[attr-defined]
                    raise ValueError(f"Semantic sequence violations:\n{report}")

                # Show warnings if allowed
                if warning_violations and validation_config.allow_semantic_warnings:
                    invariant_validator = cast(
                        InvariantValidator,
                        run_sequence._invariant_validator,
                    )
                    report = invariant_validator.generate_report(
                        warning_violations
                    )
                    logger.warning(
                        "⚠️  Semantic sequence warnings:\n%s",
                        report,
                    )

    # Pre-execution invariant validation (if enabled)
    if validation_config.validate_invariants:
        try:
            run_sequence._invariant_validator.validate_and_raise(  # type: ignore[attr-defined]
                G, validation_config.min_severity
            )
        except Exception:
            # If validation fails, provide context but don't block if it's just warnings
            if validation_config.min_severity != InvariantSeverity.WARNING:
                raise

    for op in ops_list:
        # Mark last operator for tracking (for Invariant 1)
        if not hasattr(G, "_last_operator_applied"):
            G._last_operator_applied = None  # type: ignore[attr-defined]
        G._last_operator_applied = op.name  # type: ignore[attr-defined]

        op(G, node)
        if callable(compute):
            compute(G)

        # Per-step validation (expensive, only if configured)
        if validation_config.validate_each_step and validation_config.validate_invariants:
            violations = run_sequence._invariant_validator.validate_graph(  # type: ignore[attr-defined]
                G, InvariantSeverity.ERROR
            )
            if violations:
                report = run_sequence._invariant_validator.generate_report(violations)  # type: ignore[attr-defined]
                raise ValueError(f"Invariant violations after {op.name}:\n{report}")

        # ``update_epi_via_nodal_equation`` was previously invoked here to
        # recalculate the EPI value after each operator. The responsibility for
        # updating EPI now lies with the dynamics hook configured in
        # ``compute_delta_nfr`` or with external callers.

    # Post-execution invariant validation (if enabled)
    if validation_config.validate_invariants:
        try:
            run_sequence._invariant_validator.validate_and_raise(  # type: ignore[attr-defined]
                G, validation_config.min_severity
            )
        except Exception:
            # If validation fails, provide context but don't block if it's just warnings
            if validation_config.min_severity != InvariantSeverity.WARNING:
                raise
