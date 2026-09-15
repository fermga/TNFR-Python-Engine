r"""Observe memory derived by projecting fixed pure-EPI nodal diffusion.

For ``x' = -A x``, let ``R`` be the reversible partition average, ``P``
its lift, and ``Q = I - P R``. Eliminating the unresolved coordinate ``Q x``
gives, in exact arithmetic,

    y' = -R A P y + f(t) + integral_0^t K(t-s) y(s) ds,
    K(t) = R A exp(-Q A Q t) Q A P,
    f(t) = -R A exp(-Q A Q t) Q x(0).

Every coefficient comes from the existing nodal generator and partition.
The implementation samples this identity on detached vectors using the shared
approximate matrix exponential. It observes numerical residuals, not certified
error enclosures or runtime trajectories. It neither advances a graph nor
introduces an operator or a REMESH delay law. The exact proof and the P4/P5
controls are in ``theory/DERIVED_EPI_MEMORY.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .spectral_projectors import matrix_exponential
from .structural_morphism import _build_reversible_partition_geometry

__all__ = ["EpiMemorySample", "EpiMemoryObservation", "observe_epi_memory"]


@dataclass(frozen=True)
class EpiMemorySample:
    """One finite binary64 observation; arrays are detached and read-only.

    ``source_free_epi`` retains the derived kernel with zero initial hidden
    state. ``markov_epi`` omits both the kernel and the initial-state source.
    ``identity_residual`` measures the full reconstructed rate against the
    projected fine rate; it is not an upper bound on exponential error.
    """

    time: float
    kernel: np.ndarray
    initial_source: np.ndarray
    convolution: np.ndarray
    projected_epi: np.ndarray
    markov_epi: np.ndarray
    source_free_epi: np.ndarray
    projected_rate: np.ndarray
    reconstructed_rate: np.ndarray
    identity_residual: float
    relative_identity_residual: float
    fine_semigroup_residual: float
    markov_semigroup_residual: float


@dataclass(frozen=True)
class EpiMemoryObservation:
    """Fixed-geometry projected diffusion diagnostics, without live provenance.

    ``instantaneous_generator`` is the represented product ``R A P``;
    ``quotient_generator`` is independently materialized by the shared
    reversible geometry builder. Their discrepancy and projector defects are
    exposed. ``closure_within_tolerance`` is a numerical diagnostic inherited
    from that builder, never a reason to erase small memory terms.
    """

    nodes: tuple
    blocks: tuple[tuple, ...]
    projection: np.ndarray
    lift: np.ndarray
    micro_generator: np.ndarray
    instantaneous_generator: np.ndarray
    quotient_generator: np.ndarray
    hidden_projector: np.ndarray
    hidden_generator: np.ndarray
    macro_metric_weights: np.ndarray
    initial_epi: np.ndarray
    initial_macro: np.ndarray
    initial_hidden: np.ndarray
    right_inverse_residual: float
    projector_residual: float
    quotient_generator_residual: float
    generator_residual: float
    closure_within_tolerance: bool
    tolerance: float
    numerical_tolerance: float
    samples: tuple[EpiMemorySample, ...]
    scope: str


def _real_vector(values, name: str) -> np.ndarray:
    """Reject coercion of strings, booleans and complex data into a real chart."""
    try:
        raw = np.asarray(values, dtype=object)
        if raw.ndim != 1 or not raw.size:
            raise ValueError(f"{name} must be a nonempty real vector")
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in raw
        ):
            raise ValueError(f"{name} must contain real numbers only")
        result = np.array(raw, dtype=float)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real vector") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _readonly(values: np.ndarray) -> np.ndarray:
    """Use immutable backing bytes, so writeability cannot be re-enabled."""
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("EPI memory observation exceeds finite numeric range")
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def _norm(values: np.ndarray) -> float:
    result = float(np.linalg.norm(values, ord=np.inf))
    if not np.isfinite(result):
        raise ValueError("EPI memory residual exceeds finite numeric range")
    return result


def _exponential(generator: np.ndarray, time: float) -> np.ndarray:
    scaled = time * generator
    if not np.all(np.isfinite(scaled)):
        raise ValueError("EPI memory exponential exceeds finite numeric range")
    _norm(scaled)
    result = matrix_exponential(scaled)
    if not np.all(np.isfinite(result)):
        raise ValueError("EPI memory exponential exceeds finite numeric range")
    return result


def _stationary_weights(metric: np.ndarray) -> np.ndarray:
    scaled = metric / np.max(metric)
    return scaled / np.sum(scaled)


def _check_stochastic(
    flow: np.ndarray, stationary: np.ndarray, tolerance: float
) -> float:
    """Check necessary exact diffusion identities, not exponential accuracy."""
    residual = max(
        _norm(np.sum(flow, axis=1) - 1.0),
        max(0.0, -float(np.min(flow))),
        _norm(stationary @ flow - stationary),
    )
    if residual > tolerance:
        raise ValueError(
            "EPI memory exponential failed stochastic semigroup checks "
            f"(residual={residual}, numerical_tolerance={tolerance})"
        )
    return residual


def observe_epi_memory(
    graph,
    partition,
    initial_epi,
    *,
    times,
    tolerance: float = 1e-10,
    numerical_tolerance: float = 1e-10,
) -> EpiMemoryObservation:
    """Sample projected diffusion, its derived memory and omission controls.

    ``initial_epi`` is an explicit finite real scalar vector in ``tuple(graph)``
    order; stored graph EPI is not read. ``times`` is a nonempty vector of finite
    nonnegative observation times, preserved in supplied order. They are sample
    coordinates, not fitted physical parameters or integration timesteps.

    Geometry follows ``certify_epi_coarse_graining``: fixed effective symmetric
    nonnegative conductance, positive finite capacities and row strengths,
    and a complete partition reducing to a connected quotient of at least two
    blocks. This also permits directed input whose effective conductance is
    symmetric. Phase, topology changes, events and REMESH are outside scope.

    The NumPy-only exponential is the existing scaling/squaring Taylor routine.
    Reported rate and geometry residuals expose finite arithmetic discrepancies;
    a small residual alone does not establish an accuracy bound. The diagnostic
    uses dense matrices of order at most twice the number of nodes. Independently
    of the geometry ``tolerance``, ``numerical_tolerance`` bounds observed
    row-sum, positivity and stationary-weight defects of the fine and Markov
    propagators. The generator's relative constant/weighted-total defects are
    checked before sampling, including at zero time. Samples
    failing these necessary stochasticity checks are rejected; passing them
    still does not certify exponential accuracy. Both tolerances are numerical
    policies, not physical parameters.
    """
    initial = _real_vector(initial_epi, "initial_epi")
    sample_times = _real_vector(times, "times")
    numeric_tol = float(_real_vector([numerical_tolerance], "numerical_tolerance")[0])
    if numeric_tol <= 0.0:
        raise ValueError("numerical_tolerance must be positive")
    if np.any(sample_times < 0.0):
        raise ValueError("times must be nonnegative")
    geometry = _build_reversible_partition_geometry(
        graph, partition, tolerance=tolerance, label="EPI memory"
    )
    n = len(geometry.nodes)
    if initial.shape != (n,):
        raise ValueError("initial_epi must have one value per graph node")

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            r, p = geometry.projection, geometry.lift
            a = geometry.micro_generator
            q = np.eye(n) - p @ r
            ra = r @ a
            instantaneous = ra @ p
            hidden = q @ a @ q
            hidden_drive = q @ a @ p
            macro_weights = _stationary_weights(geometry.macro_metric_weights)
            fine_weights = _stationary_weights(r.T @ macro_weights)
            generator_scale = _norm(a)
            generator_residual = max(
                _norm(np.sum(a, axis=1)), _norm(fine_weights @ a)
            )
            if generator_scale > 0.0:
                generator_residual /= generator_scale
            if generator_residual > numeric_tol:
                raise ValueError(
                    "EPI memory generator failed constant/weighted-total checks"
                )
            macro_initial = r @ initial
            hidden_initial = q @ initial
            lifted_initial = p @ macro_initial

            # The lower block integrates the memory forcing along the fine
            # reference, independently of the projected rate reconstruction.
            augmented = np.block(
                [[-a, np.zeros((n, n))], [hidden_drive @ r, -hidden]]
            )
            samples = []
            for raw_time in sample_times:
                time = float(raw_time)
                flow = _exponential(augmented, time)
                hidden_flow = _exponential(-hidden, time)
                fine_flow = flow[:n, :n]
                markov_flow = _exponential(-instantaneous, time)
                fine_residual = _check_stochastic(
                    fine_flow, fine_weights, numeric_tol
                )
                markov_residual = _check_stochastic(
                    markov_flow, macro_weights, numeric_tol
                )
                fine = fine_flow @ initial
                projected = r @ fine
                source = -ra @ hidden_flow @ hidden_initial
                convolution = ra @ (flow[n:, :n] @ initial)
                direct_rate = -ra @ fine
                reconstructed = -instantaneous @ projected + source + convolution
                residual = _norm(direct_rate - reconstructed)
                scale = max(1.0, _norm(direct_rate), _norm(reconstructed))
                samples.append(
                    EpiMemorySample(
                        time=time,
                        kernel=_readonly(ra @ hidden_flow @ hidden_drive),
                        initial_source=_readonly(source),
                        convolution=_readonly(convolution),
                        projected_epi=_readonly(projected),
                        markov_epi=_readonly(markov_flow @ macro_initial),
                        source_free_epi=_readonly(r @ fine_flow @ lifted_initial),
                        projected_rate=_readonly(direct_rate),
                        reconstructed_rate=_readonly(reconstructed),
                        identity_residual=residual,
                        relative_identity_residual=residual / scale,
                        fine_semigroup_residual=fine_residual,
                        markov_semigroup_residual=markov_residual,
                    )
                )

            return EpiMemoryObservation(
                nodes=geometry.nodes,
                blocks=geometry.blocks,
                projection=_readonly(r),
                lift=_readonly(p),
                micro_generator=_readonly(a),
                instantaneous_generator=_readonly(instantaneous),
                quotient_generator=_readonly(geometry.macro_generator),
                hidden_projector=_readonly(q),
                hidden_generator=_readonly(hidden),
                macro_metric_weights=_readonly(geometry.macro_metric_weights),
                initial_epi=_readonly(initial),
                initial_macro=_readonly(macro_initial),
                initial_hidden=_readonly(hidden_initial),
                right_inverse_residual=_norm(r @ p - np.eye(len(geometry.blocks))),
                projector_residual=_norm(q @ q - q),
                quotient_generator_residual=_norm(
                    instantaneous - geometry.macro_generator
                ),
                generator_residual=generator_residual,
                closure_within_tolerance=geometry.nodal_closure_within_tolerance,
                tolerance=float(tolerance),
                numerical_tolerance=numeric_tol,
                samples=tuple(samples),
                scope=(
                    "Finite binary64 offline samples of fixed reversible "
                    "pure-EPI diffusion; exact projected-memory identities "
                    "are separate from approximate exponential evaluation. "
                    "No graph evolution, runtime provenance, REMESH equivalence, "
                    "mesh convergence or empirical correspondence is certified."
                ),
            )
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(
            "EPI memory observation exceeds finite numeric range"
        ) from exc
